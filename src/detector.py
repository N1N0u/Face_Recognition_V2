from mtcnn import MTCNN

detector = MTCNN()

def detect_faces(frame):
    return detector.detect_faces(frame)