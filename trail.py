# ...existing code...
import time
from collections import deque, defaultdict
import numpy as np
import cv2
from ultralytics import YOLO

# Configuration
TRACK_TIME = 5.0  # seconds to keep drawing the trail
# MODEL_PATH = "yolo26n.pt"
MODEL_PATH = "yolo26n-default.pt"
SKIP_FRAMES = 500  # Frames to skip between heavy processing (1 = process every frame)

# internal counters/state for skipping
_frame_idx = 0
_last_annotated = None


model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

# trails: key -> deque of (x, y, timestamp)
trails = defaultdict(deque)

# store last-known color per key (B, G, R)
colors = {}


def color_from_key(key):
    # deterministic BGR color from key (works for numeric and string keys)
    h = abs(hash(key))
    return (
        int((h * 37) % 256),  # B
        int((h * 17) % 256),  # G
        int((h * 79) % 256),  # R
    )


def sample_box_color(img, x1, y1, x2, y2):
    """Sample a few pixels along the box border on the annotated image and return median BGR."""
    H, W = img.shape[:2]
    # clamp coords
    x1i = max(0, min(W - 1, int(round(x1))))
    y1i = max(0, min(H - 1, int(round(y1))))
    x2i = max(0, min(W - 1, int(round(x2))))
    y2i = max(0, min(H - 1, int(round(y2))))
    pts = []
    # sample small offsets on the top, bottom, left and right edges and corners
    coords = [
        ((x1i + x2i) // 2, max(0, y1i + 1)),  # top center
        ((x1i + x2i) // 2, max(0, y2i - 1)),  # bottom center
        (max(0, x1i + 1), (y1i + y2i) // 2),  # left center
        (max(0, x2i - 1), (y1i + y2i) // 2),  # right center
        (min(W - 1, x1i + 2), min(H - 1, y1i + 2)),  # near top-left
        (max(0, x2i - 2), min(H - 1, y1i + 2)),  # near top-right
        (min(W - 1, x1i + 2), max(0, y2i - 2)),  # near bottom-left
        (max(0, x2i - 2), max(0, y2i - 2)),  # near bottom-right
    ]
    for cx, cy in coords:
        pts.append(img[cy, cx].astype(np.int32))
    if not pts:
        return (0, 255, 0)
    pts = np.stack(pts, axis=0)
    median = np.median(pts, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


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


# ...existing code...
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    _frame_idx += 1

    # If we're skipping this frame and have a previous annotated image, reuse it.
    # Still advance time for trail decay.
    if (
        SKIP_FRAMES > 1
        and (_frame_idx % SKIP_FRAMES) != 0
        and _last_annotated is not None
    ):
        t = time.time()
        # annotated_frame = _last_annotated.copy()
        # boxes = None
        # Skip model.track and detection updates on this frame
    else:

        t = time.time()
        results = model.track(frame, persist=True, device="cpu")
        annotated_frame = results[0].plot()

        boxes = results[0].boxes

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

        # Add current detections to their respective trails and capture box color
        for i, xy in enumerate(xyxys):
            x1, y1, x2, y2 = xy
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)

            if id_arr is not None and i < id_arr.shape[0]:
                key = f"id{int(id_arr[i])}"
            elif cls_arr is not None and i < cls_arr.shape[0]:
                key = f"cls{int(cls_arr[i])}_f{i}"
            else:
                key = f"det{i}"

            # sample the color from the annotated_frame (box has already been drawn there)
            try:
                sampled = sample_box_color(annotated_frame, x1, y1, x2, y2)
            except Exception:
                sampled = color_from_key(key)
            colors[key] = sampled

            trails[key].append((cx, cy, t))

        # Prune old points and remove empty trails
        remove_keys = []
        for key, dq in list(trails.items()):
            while dq and (t - dq[0][2]) > TRACK_TIME:
                dq.popleft()
            if not dq:
                remove_keys.append(key)
        for k in remove_keys:
            del trails[k]
            if k in colors:
                del colors[k]

        # Draw all trails using sampled/determined color
        for key, dq in trails.items():
            if len(dq) < 2:
                continue
            pts = np.array([[int(p[0]), int(p[1])] for p in dq], dtype=np.int32)
            color = colors.get(key, color_from_key(key))
            cv2.polylines(
                annotated_frame, [pts], isClosed=False, color=color, thickness=2
            )
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

    cv2.imshow("YOLO26 Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# ...existing code...
import time
from collections import deque, defaultdict
import numpy as np
import cv2
from ultralytics import YOLO

# Configuration
TRACK_TIME = 5.0  # seconds to keep drawing the trail
MODEL_PATH = "yolo26n.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)

# trails: key -> deque of (x, y, timestamp)
trails = defaultdict(deque)

# store last-known color per key (B, G, R)
colors = {}


def color_from_key(key):
    # deterministic BGR color from key (works for numeric and string keys)
    h = abs(hash(key))
    return (
        int((h * 37) % 256),  # B
        int((h * 17) % 256),  # G
        int((h * 79) % 256),  # R
    )


def sample_box_color(img, x1, y1, x2, y2):
    """Sample a few pixels along the box border on the annotated image and return median BGR."""
    H, W = img.shape[:2]
    # clamp coords
    x1i = max(0, min(W - 1, int(round(x1))))
    y1i = max(0, min(H - 1, int(round(y1))))
    x2i = max(0, min(W - 1, int(round(x2))))
    y2i = max(0, min(H - 1, int(round(y2))))
    pts = []
    # sample small offsets on the top, bottom, left and right edges and corners
    coords = [
        ((x1i + x2i) // 2, max(0, y1i + 1)),  # top center
        ((x1i + x2i) // 2, max(0, y2i - 1)),  # bottom center
        (max(0, x1i + 1), (y1i + y2i) // 2),  # left center
        (max(0, x2i - 1), (y1i + y2i) // 2),  # right center
        (min(W - 1, x1i + 2), min(H - 1, y1i + 2)),  # near top-left
        (max(0, x2i - 2), min(H - 1, y1i + 2)),  # near top-right
        (min(W - 1, x1i + 2), max(0, y2i - 2)),  # near bottom-left
        (max(0, x2i - 2), max(0, y2i - 2)),  # near bottom-right
    ]
    for cx, cy in coords:
        pts.append(img[cy, cx].astype(np.int32))
    if not pts:
        return (0, 255, 0)
    pts = np.stack(pts, axis=0)
    median = np.median(pts, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))


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


# ...existing code...
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    t = time.time()
    results = model.track(frame, persist=True, device="cpu")
    annotated_frame = results[0].plot()

    boxes = results[0].boxes

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

    # Add current detections to their respective trails and capture box color
    for i, xy in enumerate(xyxys):
        x1, y1, x2, y2 = xy
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        if id_arr is not None and i < id_arr.shape[0]:
            key = f"id{int(id_arr[i])}"
        elif cls_arr is not None and i < cls_arr.shape[0]:
            key = f"cls{int(cls_arr[i])}_f{i}"
        else:
            key = f"det{i}"

        # sample the color from the annotated_frame (box has already been drawn there)
        try:
            sampled = sample_box_color(annotated_frame, x1, y1, x2, y2)
        except Exception:
            sampled = color_from_key(key)
        colors[key] = sampled

        trails[key].append((cx, cy, t))

    # Prune old points and remove empty trails
    remove_keys = []
    for key, dq in list(trails.items()):
        while dq and (t - dq[0][2]) > TRACK_TIME:
            dq.popleft()
        if not dq:
            remove_keys.append(key)
    for k in remove_keys:
        del trails[k]
        if k in colors:
            del colors[k]

    # Draw all trails using sampled/determined color
    for key, dq in trails.items():
        if len(dq) < 2:
            continue
        pts = np.array([[int(p[0]), int(p[1])] for p in dq], dtype=np.int32)
        color = colors.get(key, color_from_key(key))
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

    cv2.imshow("YOLO26 Webcam Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
# ...existing code...
