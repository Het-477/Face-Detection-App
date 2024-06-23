import threading

import cv2
from deepface import DeepFace

reference_img = cv2.imread("ref2.jpg")
reference_img2 = cv2.imread("ref2.jpg")

choice = input("Choose the image you want to detect: (1/2)")
if choice == 1:
    reference_img = cv2.imread("ref2.jpg")
else:
    reference_img = cv2.imread("reference_4.jpg")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 580)

counter = 0

# reference_img = cv2.imread("ref2.jpg")
# reference_img2 = cv2.imread("ref2.jpg")
# reference_img3 = cv2.imread("WIN_20231029_18_57_00_Pro.jpg")
# reference_img = cv2.imread("ref2.jpg"),cv2.imread("reference_img.jpg")

face_match = False


def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img2.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False


while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face,
                                 args=(frame.copy(),)).start()
            except ValueError:
                pass
        counter += 1
        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
