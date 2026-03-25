import os
import cv2
import numpy as np

from pipeline.droidCam import start_cam
from pipeline.detector import detect_faces
from pipeline.embedder import get_embedding, process_folder
from pipeline.recognizer import is_same_person
from pipeline.capture_dataset import capture_faces

# ========================
# CONFIG
# ========================
url = "http://10.177.208.8:4747/video"
person_name = "Atef"

raw_path = f"data/raw/{person_name}"
embedding_path = f"data/embeddings/{person_name}.npy"


# ========================
# 🧠 SMART FLOW
# ========================

# if there is no data, start capturing imges
if not os.path.exists(raw_path) or len(os.listdir(raw_path)) == 0:
    print("[INFO] No dataset found --> Starting capture...")
    capture_faces(person_name, url)


# if No Embeddings then start saving
if not os.path.exists(embedding_path):
    print("[INFO] No embedding found --> Start Building Embeddings...")
    process_folder(raw_path, embedding_path)


# load embedding
print("[INFO] Loading embedding...")
known_embedding = np.load(embedding_path)


# ========================
# RUN SYSTEM
# ========================
cap = start_cam(url)

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

        try:
            emb = get_embedding(face_img)

            same, dist = is_same_person(known_embedding, emb, threshold=0.9)

            if same:
                label = f"Authorized ({dist:.2f})"
            else:
                label = f"Unauthorized ({dist:.2f})"

        except Exception as e:
            label = "Error"
            print(e)

        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Face Verification System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()