import cv2
import numpy as np
'''a=cv2.imread('z4323099476020_2542803abc89f0bb962ec850829a326c.jpg')
b=cv2.QRCodeDetector()
m=b.detectAndDecode(a)
print(m)
cv2.imshow("aha", a)
cv2.waitKey()'''
from cvzone.PoseModule import PoseDetector
a=cv2.VideoCapture(0)
detector=PoseDetector()
while True:
    b, c= a.read()
    c=detector.findPose(c)
    e,f=detector.findPosition(c, bboxWithHands=False)
    cv2.imshow('aha', c)
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
a.release()
cv2.destroyAllWindows()

