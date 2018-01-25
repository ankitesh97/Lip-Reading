#################################################
#This file is just written just to test an individual video or an image to
#see if haar cascade works and finds the correct mouth region
#################################################

import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
# mouth_cascade = cv2.CascadeClassifier('Mouth.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

cap = cv2.VideoCapture('data4.mp4')
ctr = 0
fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Be sure to use lower case
out = cv2.VideoWriter('output.avi', fourcc, 25, (72, 43))
while True:
    print ctr
    ret, frame = cap.read()
    tempVar = frame.shape[0]
    frame = frame[int(tempVar/2.0):tempVar,:]
    ctr+=1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7,5)
    for (x,y,w,h) in mouth_rects:
        if w==43 and h==72:
            print x , y , w, h
            y = int(y - 0.15*h)
            cv2.rectangle(frame, (x,y), (x+h,y+w), (0,255,0), 3)
        # breal
    print mouth_rects
    # x,y,w,h = mouth_rects[0][0],mouth_rects[0][1],mouth_rects[0][2],mouth_rects[0][3]
    # y = int(y - 0.15*h)
    # cropped_img = frame[y:y+w,x:x+h]
    cv2.imshow('Mouth Detector', frame)
    # out.write(cropped_img)
    c = cv2.waitKey(0)
    if ctr == 29:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
