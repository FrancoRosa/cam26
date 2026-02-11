import time
from collections import deque, defaultdict
import argparse
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import ui

# Configuration
TRACK_TIME = 5.0  # seconds to keep drawing the trail
MODEL_PATH = "yolo26n.pt"

# CLI
parser = argparse.ArgumentParser(description="Fullscreen YOLO26 tracker")
parser.add_argument(
    "--device-id", type=int, default=0, help="Camera device id (default: 0)"
)
parser.add_argument(
    "--video-input",
    type=str,
    default=None,
    help="Path to video file to use instead of camera",
)
parser.add_argument(
    "--frame-period",
    type=int,
    default=4,
    help="Run inference every FRAME_PERIOD frames (default: 4)",
)
parser.add_argument(
    "--classes",
    "-c",
    type=str,
    default="",
    help="Comma-separated class names to follow (e.g. person,car). Empty = all classes",
)
args = parser.parse_args()

FRAME_PERIOD = max(1, int(args.frame_period))

model = YOLO(MODEL_PATH)

# Determine allowed class IDs from provided names (if any)
_allowed_class_ids = None
if args.classes:
    names_map = None
    try:
        # ultralytics YOLO provides model.names or names
        names_map = getattr(model, 'names', None) or getattr(model, 'model', None) and getattr(model.model, 'names', None)
    except Exception:
        names_map = None
    if isinstance(names_map, dict):
        # names_map: id->name
        provided = [s.strip() for s in args.classes.split(',') if s.strip()]
        allowed = set()
        unknown = []
        lower_map = {v.lower(): k for k, v in names_map.items()}
        for pn in provided:
            pid = lower_map.get(pn.lower())
            if pid is None:
                unknown.append(pn)
            else:
                allowed.add(pid)
        if unknown:
            print(f"Warning: unknown class names: {unknown}")
        if allowed:
            _allowed_class_ids = allowed
            print(f"Following classes (ids): {_allowed_class_ids}")
        else:
            print("No valid class names provided; following all classes.")
    else:
        print("Could not resolve model class names; --classes ignored.")

# Select video source: prefer --video-input if provided, otherwise use device id
if args.video_input:
    source = args.video_input
else:
    source = int(args.device_id)

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"ERROR: cannot open video source: {source}")
    sys.exit(1)

# trails: key -> deque of (x, y, timestamp)
trails = defaultdict(deque)

# internal frame counter and cache for last annotated image
_frame_idx = 0
_last_annotated = None


def color_from_key(key):
    # deterministic BGR color from key (works for numeric and string keys)
    h = abs(hash(key))
    return (
        int((h * 37) % 256),  # B
        int((h * 17) % 256),  # G
        int((h * 79) % 256),  # R
    )


def to_numpy_array(x):
    """Convert torch tensor / list / scalar to a numpy array, at least 1-D."""
    if x is None:
        return None
    try:
        import torch

        if isinstance(x, torch.Tensor):
            arr = x.cpu().numpy()
        else:
            arr = np.array(x)
    except Exception:
        arr = np.array(x)
    arr = np.atleast_1d(arr)
    return arr


window_name = "YOLO26 Detection"
# Resolve model class names for UI toggles
names_map = None
try:
    names_map = getattr(model, 'names', None) or (getattr(model, 'model', None) and getattr(model.model, 'names', None))
except Exception:
    names_map = None

class_names_list = []
initial_enabled = None
if isinstance(names_map, dict):
    # create ordered list by id
    class_names_list = [names_map[i] for i in sorted(names_map.keys())]
    if _allowed_class_ids is not None:
        initial_enabled = [names_map[i] for i in sorted(names_map.keys()) if i in _allowed_class_ids]

# initialize UI (creates window, fullscreen and mouse callback) with class toggles
ui.init_ui(window_name, button_w=115, button_h=50, margin=20, spacing=8, class_names=class_names_list, initial_enabled=initial_enabled)

# main loop (existing loop body remains, but replace the display step with resizing + button)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    t = time.time()
    # decide whether to run inference on this frame
    process_now = (_frame_idx % FRAME_PERIOD) == 0 or (_last_annotated is None)

    if process_now:
        results = model.track(frame, persist=True, device="cpu")
        annotated_frame = results[0].plot()
        boxes = results[0].boxes
        _last_annotated = annotated_frame.copy()

        # Safely extract arrays
        xyxys = to_numpy_array(getattr(boxes, "xyxy", None))
        if xyxys is None:
            xyxys = np.zeros((0, 4))
        else:
            # ensure shape is (N,4)
            try:
                xyxys = xyxys.reshape(-1, 4)
            except Exception:
                xyxys = np.atleast_2d(xyxys)

        cls_arr = to_numpy_array(getattr(boxes, "cls", None))
        id_arr = to_numpy_array(getattr(boxes, "id", None))

        # Add current detections to their respective trails
        for i, xy in enumerate(xyxys):
            x1, y1, x2, y2 = xy
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            # determine class id/name for this detection (if available)
            cls_id = None
            cls_name = None
            if cls_arr is not None and i < cls_arr.shape[0]:
                try:
                    cls_id = int(cls_arr[i])
                except Exception:
                    cls_id = None
                if isinstance(names_map, dict) and cls_id is not None:
                    cls_name = names_map.get(cls_id)

            # build the trail key (prefer persistent track id if available)
            if id_arr is not None and i < id_arr.shape[0]:
                key = f"id{int(id_arr[i])}"
            elif cls_name is not None:
                # use class name with frame index so different detections of same class are separate
                key = f"{cls_name}_f{i}"
            elif cls_arr is not None and i < cls_arr.shape[0]:
                key = f"cls{int(cls_arr[i])}_f{i}"
            else:
                key = f"det{i}"

            # Filtering: respect CLI --classes (by id) and UI class toggles (by name)
            allowed_by_cli = (_allowed_class_ids is None) or (cls_id is not None and cls_id in _allowed_class_ids)
            # UI toggles: if UI has class names, only follow those enabled; otherwise allow all
            allowed_by_ui = True
            try:
                enabled_names = set(ui.get_enabled_class_names())
            except Exception:
                enabled_names = set()
            if enabled_names:
                # if we have a class name for this detection, require it to be enabled
                allowed_by_ui = (cls_name is not None and cls_name in enabled_names)

            if allowed_by_cli and allowed_by_ui:
                trails[key].append((cx, cy, t))
    else:
        # reuse last annotated frame to avoid running inference
        annotated_frame = _last_annotated.copy()
        # boxes is None on skipped frames; no new detections added

    # Prune old points and remove empty trails
    remove_keys = []
    for key, dq in list(trails.items()):
        while dq and (t - dq[0][2]) > TRACK_TIME:
            dq.popleft()
        if not dq:
            remove_keys.append(key)
    for k in remove_keys:
        del trails[k]

    # Draw all trails
    for key, dq in trails.items():
        if len(dq) < 2:
            continue
        pts = np.array([[int(p[0]), int(p[1])] for p in dq], dtype=np.int32)
        color = color_from_key(key)
        cv2.polylines(annotated_frame, [pts], isClosed=False, color=color, thickness=2)
        # draw points (newest larger)
        for i, (x, y, ts) in enumerate(dq):
            pos = (int(x), int(y))
            radius = 2 if i < len(dq) - 1 else 4
            cv2.circle(annotated_frame, pos, radius, color, -1)
        # draw label near newest point
        nx, ny, _ = dq[-1]
        cv2.putText(
            annotated_frame,
            key,
            (int(nx) + 6, int(ny) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Display selected classes on the top-left of the screen
    if _allowed_class_ids is not None:
        class_labels = [f"{int(cid)}" for cid in _allowed_class_ids]
        cv2.putText(
            annotated_frame,
            "Classes: " + ", ".join(class_labels),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # use UI module to render fullscreen with buttons
    display_frame = ui.draw_ui(annotated_frame, button_w=115, button_h=50, margin=20, spacing=8)

    cv2.imshow(window_name, display_frame)

    # respond to UI Exit button
    if ui.is_exit_requested():
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    _frame_idx += 1

cap.release()
cv2.destroyAllWindows()
