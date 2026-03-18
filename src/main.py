import cv2
from droidCam import start_cam
from detector import detect_faces
from embedder import get_embedding

url = "http://10.174.9.160:4747/video"

cap = start_cam(url)

known_embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        face_img = frame[y:y + h, x:x + w]

        try:
            emb = get_embedding(face_img)

            if known_embedding is None:
                known_embedding = emb
                label = "Registered"
            else:
                from recognizer import is_same_person

                if is_same_person(known_embedding, emb):
                    label = "Same Person"
                else:
                    label = "Unknown"

            cv2.putText(
                frame,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        except:
            pass

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition V2", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()