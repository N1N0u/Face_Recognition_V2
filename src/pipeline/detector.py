from mtcnn import MTCNN
#instance from mtcnn
detector = MTCNN()

#detection function
def detect_faces(frame):
    #return result of detected face
    return detector.detect_faces(frame)