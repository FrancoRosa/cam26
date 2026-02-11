import time
from collections import deque, defaultdict
import argparse
import sys
import numpy as np
import cv2
from ultralytics import YOLO
import ui
import os
import glob

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
parser.add_argument(
    "--resolution",
    "-R",
    type=str,
    default="HD",
    help="Resolution preset: QVGA,VGA,SVGA,XGA,HD or custom WxH like 640x480 (default: HD)",
)
args = parser.parse_args()

# Resolution presets
RES_PRESETS = {
    "QVGA": (320, 240),
    "VGA": (640, 480),
    "SVGA": (800, 600),
    "XGA": (1024, 768),
    "HD": (1280, 720),
}
res_input = (args.resolution or "HD").upper()
if "X" in res_input and any(ch.isdigit() for ch in res_input):
    # allow formats like 640x480 or 640X480
    try:
        parts = res_input.replace("x", "X").split("X")
        RES_W, RES_H = int(parts[0]), int(parts[1])
    except Exception:
        RES_W, RES_H = RES_PRESETS.get("HD")
else:
    RES_W, RES_H = RES_PRESETS.get(res_input, RES_PRESETS.get("HD"))

FRAME_PERIOD = max(1, int(args.frame_period))

model = YOLO(MODEL_PATH)

# Determine allowed class IDs from provided names (if any)
_allowed_class_ids = None
if args.classes:
    names_map = None
    try:
        # ultralytics YOLO provides model.names or names
        names_map = (
            getattr(model, "names", None)
            or getattr(model, "model", None)
            and getattr(model.model, "names", None)
        )
    except Exception:
        names_map = None
    if isinstance(names_map, dict):
        # names_map: id->name
        provided = [s.strip() for s in args.classes.split(",") if s.strip()]
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


def find_preferred_camera(preferred_keywords):
    """Return the /dev/videoX path for the first device whose sysfs name
    contains any of the preferred_keywords (case-insensitive), or None.
    """
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            base = os.path.basename(dev)  # e.g. video0
            sysf = f"/sys/class/video4linux/{base}/name"
            if os.path.exists(sysf):
                with open(sysf, "r", encoding="utf-8", errors="ignore") as f:
                    name = f.read().strip()
                if not name:
                    continue
                lname = name.lower()
                for kw in preferred_keywords:
                    if kw.lower() in lname:
                        return dev
        except Exception:
            continue
    return None


# Select video source: prefer --video-input if provided, otherwise use device id
if args.video_input:
    source = args.video_input
else:
    # if user asked for device 0, try to pick the Insta360 camera if present
    if int(args.device_id) == 0:
        preferred = find_preferred_camera(
            ["insta360", "insta", "insta x4", "insta x3", "insta x"]
        )
        if preferred:
            source = preferred
            print(f"Using preferred camera device: {source}")
        else:
            source = int(args.device_id)
    else:
        source = int(args.device_id)

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"ERROR: cannot open video source: {source}")
    sys.exit(1)

# If using a camera (numeric source) attempt to set resolution to the requested preset
try:
    if args.video_input is None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
        print(f"Camera requested resolution: {RES_W}x{RES_H}")
except Exception:
    pass

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
# Define which class names to show as toggles (keep short like direct3)
TARGET_CLASS_NAMES = ["person", "car", "motorbike", "bus", "truck", "phone", "wallet"]
# Resolve model class names for UI toggles
names_map = None
try:
    names_map = getattr(model, "names", None) or (
        getattr(model, "model", None) and getattr(model.model, "names", None)
    )
except Exception:
    names_map = None

class_names_list = ["person", "car", "motorbike", "bus", "truck", "phone", "wallet"]
initial_enabled = None
if isinstance(names_map, dict):
    # map id->name exists
    name_values = {v for k, v in names_map.items()}
    # keep only the small set defined in TARGET_CLASS_NAMES and present in the model
    class_names_list = [n for n in TARGET_CLASS_NAMES if n in name_values]
    if not class_names_list:
        # fallback to all model names if none of the TARGET_CLASS_NAMES are present
        class_names_list = [names_map[i] for i in sorted(names_map.keys())]

    # prepare name->id map for initial_enabled resolution
    name_to_id = {v: k for k, v in names_map.items()}
    if _allowed_class_ids is not None:
        # enable only those within the CLI-provided allowed ids
        initial_enabled = [
            n for n in class_names_list if name_to_id.get(n) in _allowed_class_ids
        ]
    else:
        initial_enabled = list(class_names_list)

# initialize UI (creates window, fullscreen and mouse callback) with class toggles
ui.init_ui(
    window_name,
    button_w=115,
    button_h=50,
    margin=20,
    spacing=8,
    class_names=class_names_list,
    initial_enabled=initial_enabled,
)

# main loop (existing loop body remains, but replace the display step with resizing + button)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    t = time.time()
    # decide whether to run inference on this frame
    process_now = (_frame_idx % FRAME_PERIOD) == 0 or (_last_annotated is None)

    # get current UI mode and flip flag
    current_mode = ui.get_selected_mode()
    flip_top = ui.is_flip_enabled()

    if process_now:
        # Run model and get raw boxes (don't use results[0].plot() so we can selectively draw boxes)
        results = model.track(frame, persist=True, device="cpu")
        boxes = results[0].boxes

        # Safely extract arrays
        xyxys = to_numpy_array(getattr(boxes, "xyxy", None))
        if xyxys is None:
            xyxys = np.zeros((0, 4))
        else:
            try:
                xyxys = xyxys.reshape(-1, 4)
            except Exception:
                xyxys = np.atleast_2d(xyxys)

        cls_arr = to_numpy_array(getattr(boxes, "cls", None))
        id_arr = to_numpy_array(getattr(boxes, "id", None))

        # Prepare display canvas based on current_mode (0=Full,1=Top,2=Bottom,3=Side,4=Exit)
        h, w = frame.shape[:2]
        h_half = h // 2
        top_half = frame[0:h_half, :].copy()
        bottom_half = frame[h_half:h, :].copy()

        if current_mode == 0:
            display_canvas = frame.copy()
        elif current_mode == 1:
            # Top
            if flip_top:
                display_canvas = cv2.flip(top_half, 1)
            else:
                display_canvas = top_half.copy()
        elif current_mode == 2:
            # Bottom
            if flip_top:
                display_canvas = bottom_half.copy()
            else:
                display_canvas = cv2.flip(bottom_half, 1)
        elif current_mode == 3:
            # Side: concat top and bottom side by side (apply mirror as above)
            t = cv2.flip(top_half, 1) if flip_top else top_half.copy()
            b = cv2.flip(bottom_half, 1) if flip_top else bottom_half.copy()
            display_canvas = np.concatenate((t, b), axis=1)
        elif current_mode == 4:
            break

        annotated_frame = display_canvas.copy()

        try:
            enabled_names = set(ui.get_enabled_class_names())
        except Exception:
            enabled_names = set()

        # Combine drawing and trail appending with coordinate mapping depending on mode
        h, w = frame.shape[:2]
        h_half = h // 2

        for i, xy in enumerate(xyxys):
            x1, y1, x2, y2 = xy
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cls_id = None
            cls_name = None
            if cls_arr is not None and i < cls_arr.shape[0]:
                try:
                    cls_id = int(cls_arr[i])
                except Exception:
                    cls_id = None
                if isinstance(names_map, dict) and cls_id is not None:
                    cls_name = names_map.get(cls_id)

            # Filtering: respect CLI --classes (by id) and UI class toggles (by name)
            allowed_by_cli = (_allowed_class_ids is None) or (
                cls_id is not None and cls_id in _allowed_class_ids
            )
            allowed_by_ui = True
            if enabled_names:
                allowed_by_ui = cls_name is not None and cls_name in enabled_names

            if not (allowed_by_cli and allowed_by_ui):
                continue

            # Determine if detection is in top or bottom half
            is_top = cy < h_half

            # Map coordinates into display_canvas depending on mode and flip flag
            mx1 = mx2 = my1 = my2 = None
            if current_mode == 0:
                # Full: coordinates unchanged
                mx1, my1, mx2, my2 = x1, y1, x2, y2
            elif current_mode == 1:
                # Top: only keep top-half detections
                if not is_top:
                    continue
                if flip_top:
                    # top was flipped horizontally
                    mx1 = w - x2
                    mx2 = w - x1
                else:
                    mx1, mx2 = x1, x2
                my1, my2 = y1, y2
            elif current_mode == 2:
                # Bottom: only keep bottom-half detections, shift y
                if is_top:
                    continue
                # bottom flip is opposite of flip_top
                flip_bottom = not flip_top
                if flip_bottom:
                    mx1 = w - x2
                    mx2 = w - x1
                else:
                    mx1, mx2 = x1, x2
                my1 = y1 - h_half
                my2 = y2 - h_half
            elif current_mode == 3:
                # Side: left=top, right=bottom
                if is_top:
                    # map to left half
                    if flip_top:
                        mx1 = w - x2
                        mx2 = w - x1
                    else:
                        mx1, mx2 = x1, x2
                    my1, my2 = y1, y2
                    # left half has no x offset
                else:
                    # bottom -> right half; map y and x
                    flip_bottom = not flip_top
                    if flip_bottom:
                        mx1 = w - x2
                        mx2 = w - x1
                    else:
                        mx1, mx2 = x1, x2
                    my1 = y1 - h_half
                    my2 = y2 - h_half
                    # shift to right half by adding w
                    mx1 += w
                    mx2 += w
            else:
                # Exit or unknown
                continue

            # Round and clamp
            try:
                ix1, iy1, ix2, iy2 = (
                    int(round(mx1)),
                    int(round(my1)),
                    int(round(mx2)),
                    int(round(my2)),
                )
            except Exception:
                continue
            ih, iw = annotated_frame.shape[:2]
            ix1 = max(0, min(ix1, iw - 1))
            ix2 = max(0, min(ix2, iw - 1))
            iy1 = max(0, min(iy1, ih - 1))
            iy2 = max(0, min(iy2, ih - 1))

            # draw box and label
            col_key = (
                cls_name
                if cls_name is not None
                else (cls_id if cls_id is not None else i)
            )
            color = color_from_key(col_key)
            cv2.rectangle(annotated_frame, (ix1, iy1), (ix2, iy2), color, 2)
            label = (
                cls_name
                if cls_name is not None
                else (str(cls_id) if cls_id is not None else "obj")
            )
            cv2.putText(
                annotated_frame,
                label,
                (ix1, max(10, iy1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            # Append trail point in display coordinates
            mcx = (ix1 + ix2) / 2.0
            mcy = (iy1 + iy2) / 2.0
            if id_arr is not None and i < id_arr.shape[0]:
                key = f"id{int(id_arr[i])}"
            elif cls_name is not None:
                key = f"{cls_name}_f{i}"
            elif cls_arr is not None and i < cls_arr.shape[0]:
                key = f"cls{int(cls_arr[i])}_f{i}"
            else:
                key = f"det{i}"

            trails[key].append((mcx, mcy, t))
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
    display_frame = ui.draw_ui(
        annotated_frame, button_w=115, button_h=50, margin=20, spacing=8
    )

    cv2.imshow(window_name, display_frame)

    # respond to UI Exit button
    if ui.is_exit_requested():
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    _frame_idx += 1


def cleanup():
    """Release camera/device and model gracefully and destroy windows."""
    try:
        if "cap" in globals() and cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    except Exception:
        pass
    # try model close/release if available
    try:
        if "model" in globals() and model is not None:
            for fn in ("close", "release"):
                f = getattr(model, fn, None)
                if callable(f):
                    try:
                        f()
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


cleanup()
