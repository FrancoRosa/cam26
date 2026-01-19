import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)
    if not cap.isOpened():
        continue

    print(f"Showing camera ID {i}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow(f"Camera {i}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
