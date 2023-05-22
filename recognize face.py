import cv2
import numpy as np
from PIL import Image
import os
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
facescascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font=cv2.FONT_HERSHEY_SIMPLEX
id=0
names=['loiuy','kimmich','a', 'ola', 'hjui', 'tyer']
cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(4,480)
min=0.1*cam.get(3)
max=0.1*cam.get(4)
while (1):
    ret,img=cam.read()

    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces=facescascade.detectMultiScale(gray,scaleFactor=1.2, minNeighbors=5,minSize=(int(min),int(min)),)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h), (0,255,0),2)
        id, confidence= recognizer.predict(gray[y:y+h,x:x+w])
        if (confidence<100):
            id=names[id]
            confidence=" {0}%".format(round(100-confidence))
        else:
            id="tuấn"
            confidence = " {0}%".format(round(100 - confidence))
        cv2.putText(img,str(id),(x+5,y-5),font,1,(255,255,255),2)
        cv2.putText(img, str(confidence), (x + 5, y+h - 5), font, 1, (255, 255, 255), 2)
    cv2.imshow('nhận diện faces', img)
    if cv2.waitKey(10) == ord('q'):
        break
print('exit')
cam.release()
cv2.destroyAllWindows()


