import cv2

cap = cv2.VideoCapture("http://10.174.9.160:4747/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("TEST WINDOW", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()