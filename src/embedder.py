from keras_facenet import FaceNet

embedder = FaceNet()

def get_embedding(face_img):
    return embedder.embeddings([face_img])[0]