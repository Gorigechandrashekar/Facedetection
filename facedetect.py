import cv2 as cv
import numpy as np

capture=cv.VideoCapture(0)# captures video from webcam

while(True):
    isTrue,frame=capture.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY) #converts every frame into grayscale image to detect faces
    haar_cascade=cv.CascadeClassifier('haar_face.xml') #imported haarcascade code
    faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3) #detects faces

    for x,y,w,h in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2) #adds rectangle-border over the face

    cv.imshow(" face detection",frame) #shows the result in webframe

    if cv.waitKey(20) & 0xFF==ord('d'):
        break

print(len(faces_rect)) #finally after interruption of keyboard prints how many faces detected 


cv.waitKey(0)
cv.destroyAllWindows()
