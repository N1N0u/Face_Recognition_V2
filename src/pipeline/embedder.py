import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from numpy.linalg import norm

embedder = FaceNet()

def get_embedding(face_img):
    if face_img is None or face_img.size == 0:
        raise ValueError("Invalid face image")

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32') / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    embedding = embedder.embeddings(face_img)[0]
    embedding = embedding / norm(embedding)

    return embedding



def process_folder(input_dir, output_path):
    embeddings = []

    print(f"[INFO] Processing folder: {input_dir}")

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            emb = get_embedding(img)
            embeddings.append(emb)

            print(f"[OK] {img_name}")

        except Exception as e:
            print(f"[ERROR] {img_name}: {e}")

    if not embeddings:
        raise ValueError("No valid images found")

    embeddings = np.array(embeddings)


    mean_embedding = embeddings.mean(axis=0)


    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, mean_embedding)

    print(f"[INFO] Saved embedding → {output_path}")

    return mean_embedding