import os
import math
import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')

SOURCE_DIRECTORY = 'original/'
TARGET_DIRECTORY = 'modified/'
words = os.listdir(SOURCE_DIRECTORY)
bakwasFile = open('bakwasFile.txt','w')
for word in words:
    os.mkdir(TARGET_DIRECTORY+word)
    for setType in ['/test/','/train/','/val/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        os.mkdir(TARGET_DIRECTORY+word+setType)
        for files in testSet:
            if files.endswith('.mp4'):
                videoFileName = files
                metaFileName = files.replace('mp4','txt')
                metaFile = open(SOURCE_DIRECTORY + word + setType + metaFileName)
                fileSeconds = float(metaFile.readlines()[4].split()[1])
                fileFrames = math.ceil(fileSeconds*25)
                if fileFrames%2==0:
                    fileFrames+=1
                frameRangeStart = int(15- ((fileFrames-1)/2)-1)
                frameRangeEnd = int(15 + ((fileFrames-1)/2)-1)
                print fileFrames,frameRangeStart,frameRangeEnd,videoFileName,setType
                cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(TARGET_DIRECTORY+word+setType+videoFileName.replace('mp4','avi'),fourcc,1,(72, 43))
                ctr = 0
                default_mouth_rects = []
                bakwasFileFlag = False
                # print '=======First Loop Starts=============='
                while(ctr<29):
                    ret, frame = cap.read()
                    tempVar = frame.shape[0]
                    frame = frame[int(tempVar/2.0):tempVar,:]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
                    if len(mouth_rects)!=0:
                        default_mouth_rects = mouth_rects
                        print 'I got mouth rects in frame ',ctr
                        break
                    if ctr==28 and len(mouth_rects)==0:
                        bakwasFile.write(TARGET_DIRECTORY+word+setType+videoFileName.replace('mp4','avi')+'\n')
                        bakwasFileFlag = True
                    ctr+=1
                # print '=======First Loop Ends=============='
                if bakwasFileFlag ==False:
                    ctr = 0
                    previous_mouth_rects = default_mouth_rects
                    cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
                    mouth_rects=[]
                    # print '=======Second Loop Starts=============='
                    while(ctr<29):
                        # print ctr
                        ret, frame = cap.read()
                        tempVar = frame.shape[0]
                        if (frameRangeStart<=ctr and ctr<=frameRangeEnd):
                            frame = frame[int(tempVar/2.0):tempVar,:]
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if len(mouth_rects)!=0:
                                previous_mouth_rects = mouth_rects
                            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 11)
                            try:
                                x,y,w,h = mouth_rects[0][0],mouth_rects[0][1],mouth_rects[0][2],mouth_rects[0][3]
                                y = max(int(y - 0.15*h),0)
                            except :
                                # print ctr, "Previous Liya"
                                x,y,w,h = previous_mouth_rects[0][0],previous_mouth_rects[0][1],previous_mouth_rects[0][2],previous_mouth_rects[0][3]
                                ctr+=1
                                continue
                            cropped_img = frame[y:y+w,x:x+h]
                            out.write(cropped_img)
                        ctr+=1
                    # print '=======Second Loop Ends=============='
