import cv2
import numpy as np
import argparse
from yolo26 import setup_model, post_process, IMG_SIZE, CLASSES
from colors import colors

# --- Global variables for UI and mode selection ---
current_mode = 0
flip_top_half = False
modes = {0: "Full", 1: "Top", 2: "Bottom", 3: "Side", 4: "Exit"}
tracker_ages = [50, 100, 150]
tracker_age_labels = ["Min", "Mid", "Max"]
current_tracker_age_idx = 0
button_width = 90
button_height = 40
button_gap = 20
start_y = 20

webcam_width = 1280
webcam_height = 720
SCREEN_HEIGHT = 1080
SCREEN_WIDTH = 1920
DEFAULT_CAM = 0
# --- Class Definitions and Colors ---
TARGET_CLASS_NAMES = ["person", "car", "motorbike", "bus", "truck", "pile"]

# TARGET_CLASS_NAMES = [
#     "person", "bicycle", "car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
#            "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
#            "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
#            "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife",
#            "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
#            "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
#            "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier", "toothbrush"
# ]

TARGET_CLASSES_MAP = {name: i for i, name in enumerate(CLASSES)}
TARGET_CLASS_IDS = [TARGET_CLASSES_MAP[name] for name in TARGET_CLASS_NAMES]
CLASS_TOGGLES = {cls: True for cls in TARGET_CLASS_NAMES}
np.random.seed(0)

CLASS_COLORS = {
    cls: (int(c[0]), int(c[1]), int(c[2]))
    for cls, c in zip(
        TARGET_CLASS_NAMES, np.random.randint(0, 255, size=(len(TARGET_CLASS_NAMES), 3))
    )
}

# --- Mouse Callback (Unchanged) ---


def get_track_color(track_id):
    """Generate a unique color for each track id."""
    np.random.seed(track_id)
    color = tuple(int(x) for x in np.random.randint(64, 255, size=3))
    return color


def mouse_callback(event, x, y, flags, param):
    global current_mode, CLASS_TOGGLES, tracker
    global current_tracker_age_idx, tracker_ages
    global flip_top_half

    if event == cv2.EVENT_LBUTTONDOWN:
        # Mode buttons
        for mode_id in range(4):
            start_x = 10 + (button_width + button_gap) * mode_id
            if (
                start_x < x < start_x + button_width
                and start_y < y < start_y + button_height
            ):
                current_mode = mode_id
                print(f"Mode changed to: {modes[current_mode]}")
                return

        # Clear Tracks Button
        clear_button_x = (
            10 + (button_width + button_gap) * 3 + button_width + button_gap
        )
        clear_button_y = start_y
        if (
            clear_button_x < x < clear_button_x + button_width
            and clear_button_y < y < clear_button_y + button_height
        ):
            tracker.tracks = {}
            tracker.next_id = 0
            print("Tracking lines cleared.")
            return
        # Exit button
        exit_button_x, exit_button_y = 10, args.height - button_height - 10
        if (
            exit_button_x < x < exit_button_x + button_width
            and exit_button_y < y < exit_button_y + button_height
        ):
            current_mode = 4
            print("Exit button clicked.")

        # Tracker Age Button
        tracker_age_x = (
            10
            + (button_width + button_gap) * 3
            + button_width
            + button_gap
            + button_width
            + button_gap
        )
        tracker_age_y = start_y
        if (
            tracker_age_x < x < tracker_age_x + button_width
            and tracker_age_y < y < tracker_age_y + button_height
        ):
            current_tracker_age_idx = (current_tracker_age_idx + 1) % 3
            tracker.max_age = tracker_ages[current_tracker_age_idx]
            print(f"Tracker age set to {tracker.max_age}")
            return

        # --- Flip Top Half Button (top-right corner) ---
        btn_w, btn_h = 120, 40
        btn_x = args.width - btn_w - 20  # Use args.width for window width
        btn_y = 20
        if btn_x < x < btn_x + btn_w and btn_y < y < btn_y + btn_h:
            flip_top_half = not flip_top_half
            print(f"Flip top half: {flip_top_half}")
            return
        # Class toggles
        y_start_cls = 60
        for i, cls in enumerate(TARGET_CLASS_NAMES):
            y_cls = y_start_cls + i * 25
            if 10 < x < 30 and y_cls < y < y_cls + 20:
                CLASS_TOGGLES[cls] = not CLASS_TOGGLES[cls]
                print(
                    f"Toggled class '{cls}': {'Enabled' if CLASS_TOGGLES[cls] else 'Disabled'}"
                )
                return


# --- SIMPLIFIED: Simple Tracker ---
# This tracker works directly with the coordinates of the current canvas.


class SimpleTracker:
    def __init__(self, max_age=150):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age

    def update(self, boxes):
        detected_centers = [
            ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in boxes
        ]
        unmatched_detections = list(range(len(detected_centers)))
        matched_track_ids = set()

        for tid, track in self.tracks.items():
            if not track["positions"]:
                continue
            last_pos = track["positions"][-1]
            best_match_idx, min_dist = -1, 75  # Distance threshold
            for i in unmatched_detections:
                dist = np.hypot(
                    detected_centers[i][0] - last_pos[0],
                    detected_centers[i][1] - last_pos[1],
                )
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                track["positions"].append(detected_centers[best_match_idx])
                track["missed_frames"] = 0
                matched_track_ids.add(tid)
                unmatched_detections.remove(best_match_idx)

        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]["missed_frames"] += 1
                if self.tracks[tid]["missed_frames"] >= self.max_age:
                    del self.tracks[tid]

        for i in unmatched_detections:
            self.tracks[self.next_id] = {
                "positions": [detected_centers[i]],
                "missed_frames": 0,
            }
            self.next_id += 1

    def draw_tracks(self, img):
        for tid, track_data in self.tracks.items():
            positions = track_data["positions"]
            if len(positions) > 1:
                color = get_track_color(tid)
                cv2.polylines(
                    img,
                    [np.array(positions)],
                    isClosed=False,
                    color=(0, 255, 0),
                    thickness=2,
                )
                last_pos = positions[-1]
                cv2.circle(img, last_pos, 5, color, -1)
                cv2.putText(
                    img,
                    f"ID:{tid}",
                    last_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colors.green,
                    2,
                )


# --- UI Drawing Functions (Unchanged) ---


def draw_buttons(frame):
    for mode_id, mode_name in modes.items():
        if mode_id == 4:
            continue
        start_x = 10 + (button_width + button_gap) * mode_id
        color = colors.green if mode_id == current_mode else colors.gray
        cv2.rectangle(
            frame,
            (start_x, start_y),
            (start_x + button_width, start_y + button_height),
            color,
            -1,
        )
        cv2.rectangle(
            frame,
            (start_x, start_y),
            (start_x + button_width, start_y + button_height),
            colors.white,
            2,
        )
        text_size = cv2.getTextSize(mode_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = start_x + (button_width - text_size[0]) // 2
        text_y = start_y + (button_height + text_size[1]) // 2
        cv2.putText(
            frame,
            mode_name,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colors.black if mode_id == current_mode else colors.white,
            1,
            cv2.LINE_AA,
        )

    start_x_clear, start_y_clear = (
        10 + (button_width + button_gap) * 3 + button_width + button_gap,
        start_y,
    )  # Aligned with mode buttons
    color_clear = (92, 81, 53)  # Blue color
    cv2.rectangle(
        frame,
        (start_x_clear, start_y_clear),
        (start_x_clear + button_width, start_y_clear + button_height),
        color_clear,
        -1,
    )
    cv2.rectangle(
        frame,
        (start_x_clear, start_y_clear),
        (start_x_clear + button_width, start_y_clear + button_height),
        colors.white,
        2,
    )
    text_size_clear = cv2.getTextSize("Clear", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x_clear = start_x_clear + (button_width - text_size_clear[0]) // 2
    text_y_clear = start_y_clear + (button_height + text_size_clear[1]) // 2
    cv2.putText(
        frame,
        "Clear",
        (text_x_clear, text_y_clear),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors.white,
        1,
        cv2.LINE_AA,
    )

    start_x_exit, start_y_exit = 10, frame.shape[0] - button_height - 10
    color_exit = colors.red if current_mode == 4 else (50, 50, 200)
    cv2.rectangle(
        frame,
        (start_x_exit, start_y_exit),
        (start_x_exit + button_width, start_y_exit + button_height),
        color_exit,
        -1,
    )
    cv2.rectangle(
        frame,
        (start_x_exit, start_y_exit),
        (start_x_exit + button_width, start_y_exit + button_height),
        colors.white,
        2,
    )
    text_size_exit = cv2.getTextSize(modes[4], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x_exit = start_x_exit + (button_width - text_size_exit[0]) // 2
    text_y_exit = start_y_exit + (button_height + text_size_exit[1]) // 2
    cv2.putText(
        frame,
        modes[4],
        (text_x_exit, text_y_exit),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors.white,
        1,
        cv2.LINE_AA,
    )

    # Tracker Age Button
    tracker_age_x = start_x_clear + button_width + button_gap
    tracker_age_y = start_y_clear
    color_age = (60, 120, 200)
    cv2.rectangle(
        frame,
        (tracker_age_x, tracker_age_y),
        (tracker_age_x + button_width, tracker_age_y + button_height),
        color_age,
        -1,
    )
    cv2.rectangle(
        frame,
        (tracker_age_x, tracker_age_y),
        (tracker_age_x + button_width, tracker_age_y + button_height),
        colors.white,
        2,
    )
    label = f"Age: {tracker_age_labels[current_tracker_age_idx]}"
    text_size_age = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x_age = tracker_age_x + (button_width - text_size_age[0]) // 2
    text_y_age = tracker_age_y + (button_height + text_size_age[1]) // 2
    cv2.putText(
        frame,
        label,
        (text_x_age, text_y_age),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors.white,
        1,
        cv2.LINE_AA,
    )

    # --- Flip Top Half Button (top-right corner) ---
    btn_w, btn_h = 120, 40
    btn_x = frame.shape[1] - btn_w - 20
    btn_y = 20
    color = colors.green if flip_top_half else colors.gray
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), color, -1)
    cv2.rectangle(
        frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), colors.white, 2
    )
    label = "Mirror"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = btn_x + (btn_w - text_size[0]) // 2
    text_y = btn_y + (btn_h + text_size[1]) // 2
    cv2.putText(
        frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colors.white,
        1,
        cv2.LINE_AA,
    )


def draw_class_toggles(frame):
    y_start = 80
    for i, cls in enumerate(TARGET_CLASS_NAMES):
        y_pos = y_start + i * 25
        checkbox_color = colors.green if CLASS_TOGGLES[cls] else colors.darkgreen
        cv2.rectangle(frame, (10, y_pos), (30, y_pos + 20), checkbox_color, -1)
        cv2.rectangle(frame, (10, y_pos), (30, y_pos + 20), colors.white, 2)
        cv2.putText(
            frame,
            cls,
            (40, y_pos + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            CLASS_COLORS[cls],
            1,
            cv2.LINE_AA,
        )


# --- Core Logic Functions ---


def draw_objects(frame, boxes, scores, classes):
    for box, score, cls_id in zip(boxes, scores, classes):
        cls_name = CLASSES[cls_id]
        if CLASS_TOGGLES.get(cls_name, False):
            x1, y1, x2, y2 = box
            color = CLASS_COLORS[cls_name]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls_name} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )


def run_detection(frame, model, platform):
    """
    Pads the frame to a square (letterboxing), runs detection, and then scales
    the bounding boxes back to the original frame's coordinate system.
    """
    if frame is None or frame.size == 0:
        return [], [], []

    original_h, original_w = frame.shape[:2]
    target_w, target_h = IMG_SIZE

    # Step 1: Calculate letterbox scaling and padding
    scale = min(target_w / original_w, target_h / original_h)
    new_w, new_h = int(original_w * scale), int(original_h * scale)
    pad_w, pad_h = (target_w - new_w) // 2, (target_h - new_h) // 2

    # Step 2: Resize and create a padded square image for the model
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_img = np.full(
        (target_h, target_w, 3), 128, dtype=np.uint8
    )  # Gray background
    padded_img[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized_frame

    # Step 3: Run the detection model
    img_rgb = cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB)
    input_data = (
        img_rgb.transpose((2, 0, 1)).astype(np.float32) / 255.0
        if platform in ["pytorch", "onnx"]
        else img_rgb
    )

    outputs = model.run([input_data])
    boxes, classes, scores = post_process(outputs)

    filtered_boxes, filtered_classes, filtered_scores = [], [], []
    if boxes is not None and len(boxes) > 0:
        # Step 4: Transform bounding boxes back to the original cropped frame coordinates
        for i in range(len(boxes)):
            if classes[i] in TARGET_CLASS_IDS:
                b = boxes[i]

                # Undo padding
                x1 = (b[0] - pad_w) / scale
                y1 = (b[1] - pad_h) / scale
                x2 = (b[2] - pad_w) / scale
                y2 = (b[3] - pad_h) / scale

                # Clip to the original frame boundaries
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(original_w, int(x2))
                y2 = min(original_h, int(y2))

                if x1 < x2 and y1 < y2:
                    filtered_boxes.append([x1, y1, x2, y2])
                    filtered_classes.append(classes[i])
                    filtered_scores.append(scores[i])

    return filtered_boxes, filtered_classes, filtered_scores


def main():
    global current_mode, tracker, args
    global current_tracker_age_idx, tracker_ages
    global flip_top_half  # Add this

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./yolo26n.pt")
    # parser.add_argument("--model_path", type=str, default="./yolo11n.onnx")
    parser.add_argument("--target", type=str, default="rk3576")
    parser.add_argument("--device_id", type=str, default=None)
    parser.add_argument("--camera_id", default=DEFAULT_CAM)
    parser.add_argument("--width", type=int, default=SCREEN_WIDTH)  # 1920
    parser.add_argument("--height", type=int, default=SCREEN_HEIGHT)  # 550
    args = parser.parse_args()
    background_image = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    window_name = "360safe"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, background_image)

    cap = cv2.VideoCapture(args.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, webcam_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, webcam_height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    model, platform = setup_model(args)

    # tracker = SimpleTracker()
    tracker = SimpleTracker(max_age=tracker_ages[current_tracker_age_idx])

    last_mode = -1

    # --- Add frame counter and detection interval ---
    frame_counter = 0
    detection_interval = 3  # Run detection every 3 frames
    last_boxes, last_classes, last_scores = [], [], []

    while True:
        ret, original_frame = cap.read()
        # original_frame=cv2.flip(original_frame,1)
        if not ret:
            break

        frame_counter += 1

        # Reset tracker and last results if mode changes
        if current_mode != last_mode:
            tracker = SimpleTracker()
            last_boxes, last_classes, last_scores = [], [], []
            last_mode = current_mode

        h, w = original_frame.shape[:2]
        h_half = h // 2
        top_half = original_frame[0:h_half, :]
        bottom_half = original_frame[h_half:h, :]
        if flip_top_half:
            top_half_flipped = cv2.flip(top_half, 1)  # 1 for horizontal flip
            original_frame[0:h_half, :] = top_half_flipped
        else:
            bottom_half_flipped = cv2.flip(bottom_half, 1)  # 1 for horizontal flip
            original_frame[h_half:h, :] = bottom_half_flipped
        # --- Prepare Display Canvas (every frame) ---
        display_canvas = None
        if current_mode == 0:
            display_canvas = original_frame.copy()
        elif current_mode == 1:
            display_canvas = original_frame[0:h_half, 0:w]
        elif current_mode == 2:
            display_canvas = original_frame[h_half:h, 0:w]
        elif current_mode == 3:
            top_half = original_frame[0:h_half, 0:w]
            bottom_half = original_frame[h_half:h, 0:w]
            display_canvas = np.concatenate((top_half, bottom_half), axis=1)
        elif current_mode == 4:
            break

        # --- Run Detection Periodically ---
        if frame_counter % detection_interval == 0:
            current_boxes, current_classes, current_scores = [], [], []
            if display_canvas is not None:
                current_boxes, current_classes, current_scores = run_detection(
                    display_canvas, model, platform
                )
            last_boxes = current_boxes
            last_classes = current_classes
            last_scores = current_scores
            tracker.update(last_boxes)

        # --- Update Tracker and Draw on Canvas (every frame) ---
        if display_canvas is not None:
            # Use the last known results for drawing
            draw_objects(display_canvas, last_boxes, last_scores, last_classes)
            tracker.draw_tracks(display_canvas)

            # --- Final Display Preparation ---
            final_display_frame = cv2.resize(display_canvas, (args.width, args.height))
            draw_buttons(final_display_frame)
            draw_class_toggles(final_display_frame)
            cv2.imshow(window_name, final_display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    model.release()


if __name__ == "__main__":
    main()
