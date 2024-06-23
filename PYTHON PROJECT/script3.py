from deepface.detectors import FaceDetector
import cv2

img_path = "ref2.jpg"
detector_name = "opencv"

img = cv2.imread(img_path)

detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface

obj = FaceDetector.detect_faces(detector, detector_name, img)

print("there are ",len(obj)," faces")