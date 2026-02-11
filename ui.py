import cv2
import numpy as np

# UI module for fulltrack
# Provides fullscreen display, button drawing and mouse handling for modes

MODE_LABELS = {0: "Full", 1: "Top", 2: "Bottom", 3: "Side", 4: "Exit"}
selected_mode = 0
exit_requested = False
button_rects = []  # list of tuples (x,y,w,h,mode)
window_name = None
SCREEN_W = None
SCREEN_H = None
CLASS_NAMES = []
CLASS_TOGGLES = {}


def _get_screen_size_tk():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return None, None


def init_ui(
    win_name,
    button_w=140,
    button_h=50,
    margin=20,
    spacing=10,
    class_names=None,
    initial_enabled=None,
):
    """Initialize UI: create window, compute button positions (left-bottom stacked).
    Call once before the main loop.
    class_names: list of class name strings to show toggles for (top-left)
    initial_enabled: optional list of names to enable initially (others disabled)
    """
    global window_name, SCREEN_W, SCREEN_H, button_rects, CLASS_NAMES, CLASS_TOGGLES
    window_name = win_name
    SCREEN_W, SCREEN_H = _get_screen_size_tk()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    cv2.setMouseCallback(window_name, _on_mouse)

    # setup class toggles
    if class_names is None:
        class_names = []
    CLASS_NAMES = list(class_names)
    if initial_enabled is None:
        CLASS_TOGGLES = {name: True for name in CLASS_NAMES}
    else:
        enabled_lower = {n.lower() for n in initial_enabled}
        CLASS_TOGGLES = {name: (name.lower() in enabled_lower) for name in CLASS_NAMES}

    # compute button rects based on screen size (or will be computed at draw time)
    button_rects = []
    # populate with placeholders; actual coordinates computed in draw_ui using display size
    for mode in sorted(MODE_LABELS.keys()):
        button_rects.append((margin, margin, button_w, button_h, mode))


def _on_mouse(event, x, y, flags, param):
    """Mouse callback for cv2 window; updates selected_mode / exit flag and class toggles."""
    global selected_mode, exit_requested, button_rects, CLASS_TOGGLES, CLASS_NAMES
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    # check buttons
    for bx, by, bw, bh, mode in button_rects:
        if bx <= x <= bx + bw and by <= y <= by + bh:
            selected_mode = mode
            if MODE_LABELS.get(mode, "") == "Exit":
                exit_requested = True
            return
    # check class toggles (top-left)
    y_start = 80
    for i, name in enumerate(CLASS_NAMES):
        y_pos = y_start + i * 25
        if 10 < x < 30 and y_pos < y < y_pos + 20:
            CLASS_TOGGLES[name] = not CLASS_TOGGLES.get(name, True)
            return


def draw_ui(
    annotated_frame, button_w=140, button_h=50, margin=20, spacing=10, alpha=0.35
):
    """Resize annotated_frame to fullscreen (if available) and draw left-bottom stacked buttons.
    Returns the display_frame ready for imshow.
    """
    global button_rects, CLASS_NAMES, CLASS_TOGGLES
    H, W = annotated_frame.shape[:2]
    if SCREEN_W is None or SCREEN_H is None:
        display_w, display_h = W, H
    else:
        display_w, display_h = SCREEN_W, SCREEN_H

    display_frame = cv2.resize(
        annotated_frame, (display_w, display_h), interpolation=cv2.INTER_LINEAR
    )

    # compute button positions stacked from bottom-left upwards
    button_rects = []
    x = margin
    modes = sorted(MODE_LABELS.keys())
    for i, mode in enumerate(modes):
        by = display_h - margin - (i + 1) * (button_h + spacing) + spacing
        bw = button_w
        bh = button_h
        button_rects.append((x, by, bw, bh, mode))

    overlay = display_frame.copy()
    for bx, by, bw, bh, mode in button_rects:
        color = (0, 0, 255) if mode == 4 else (0, 128, 200)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, -1)

    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

    # draw borders and text, highlight selected
    for bx, by, bw, bh, mode in button_rects:
        label = MODE_LABELS.get(mode, str(mode))
        # border
        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 0, 180), 2)
        # highlight selected
        if mode == selected_mode:
            cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        # text centered vertically
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        tx = bx + 10
        ty = by + (bh + text_size[1]) // 2
        cv2.putText(
            display_frame,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # draw class toggles at top-left
    y_start = 80
    for i, name in enumerate(CLASS_NAMES):
        y_pos = y_start + i * 25
        checkbox_color = (0, 200, 0) if CLASS_TOGGLES.get(name, True) else (30, 30, 30)
        cv2.rectangle(display_frame, (10, y_pos), (30, y_pos + 20), checkbox_color, -1)
        cv2.rectangle(display_frame, (10, y_pos), (30, y_pos + 20), (255, 255, 255), 1)
        cv2.putText(
            display_frame,
            name,
            (40, y_pos + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return display_frame


def is_exit_requested():
    return exit_requested


def get_selected_mode():
    return selected_mode


def get_enabled_class_names():
    return [name for name, enabled in CLASS_TOGGLES.items() if enabled]
