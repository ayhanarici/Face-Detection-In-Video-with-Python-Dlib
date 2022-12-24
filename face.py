import cv2
import numpy as np
from imutils import face_utils
import dlib
def findFace():
    p = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    videoFile="Çöpçüler Kralı - Fragman.mp4"
    cap = cv2.VideoCapture(videoFile)
    cap2 = cv2.VideoCapture(videoFile)
    while True:
        ret, image = cap.read()
        ret2, original = cap2.read()
        image = cv2.resize(image, (640, 360))
        original = cv2.resize(original, (640, 360))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        vertical = np.concatenate((image,original),axis=0)
        cv2.imshow("Output Video-Press Esc to Exit", vertical)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cap.release()
    cap2.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    findFace()
