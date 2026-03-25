import cv2
import os
from pipeline.droidCam import start_cam
from pipeline.detector import detect_faces

def capture_faces(person_name, url, max_images=20):
    cap = start_cam(url)

    save_path = f"data/raw/{person_name}"
    os.makedirs(save_path, exist_ok=True)

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)

            face_img = frame[y:y+h, x:x+w]

            if face_img.size == 0:
                continue


            if count % 5 == 0 and saved_count < max_images:
                file_path = f"{save_path}/{saved_count}.jpg"
                cv2.imwrite(file_path, face_img)

                saved_count += 1


                print(f"[INFO] Saved image {saved_count}/{max_images}")

            count += 1


            cv2.putText(
                frame,
                f"{saved_count}/{max_images}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Capture", frame)


        if saved_count >= max_images:
            print("[INFO] Capture completed.")
            break

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()