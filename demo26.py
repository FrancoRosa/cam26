import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Run YOLO26 inference on the frame
    # We use persist=True for better object tracking across frames
    results = model.track(frame, persist=True, device="cpu")

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow("YOLO26 Webcam Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
