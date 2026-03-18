import cv2
def start_cam(url):
    cap=cv2.VideoCapture(url)
    return cap