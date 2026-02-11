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

# create fullscreen window (works on X11; small imshow+waitKey helps on Ubuntu/Wayland)
window_name = "YOLO26 Webcam Detection"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
try:
    cv2.imshow(window_name, np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.waitKey(1)
except Exception:
    pass

# trails: key -> deque of (x, y, timestamp)
trails = defaultdict(deque)


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


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- Split top/bottom and mirror bottom (like direct3.py) ---
    try:
        h, w = frame.shape[:2]
        if h >= 2:
            h_half = h // 2
            top = frame[0:h_half, :, :]
            bottom = frame[h_half:h, :, :]
            # mirror bottom horizontally
            bottom_mirror = cv2.flip(bottom, 1)
            # compose processed frame (top + mirrored bottom)
            processed_frame = np.vstack([top, bottom_mirror])
        else:
            processed_frame = frame
    except Exception:
        # fallback to original if any issue
        processed_frame = frame

    t = time.time()

    # Run inference on the processed (mirrored bottom) frame
    results = model.track(processed_frame, persist=True, device="cpu")
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

    # Add current detections to their respective trails (coordinates already match processed_frame)
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

    # Draw all trails on the annotated_frame
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

    cv2.imshow(window_name, annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
