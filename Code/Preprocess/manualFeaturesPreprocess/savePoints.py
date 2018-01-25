#################################################
#This file is for creating the 9wordsX1000videosX(frames X 12 coordinates) in a .txt file
#stored in 9wordsX1000videosXY
#################################################
import os
import math
import cv2
import numpy as np
from imutils import face_utils
import dlib
from collections import OrderedDict
# import json
# import codecs
fielDone = 0
totalFile = 5500
SOURCE_DIRECTORY = 'original-large/'
TARGET_DIRECTORY = 'lip-border/'
words = os.listdir(SOURCE_DIRECTORY)

args={}
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 60))
])
args["shape_predictor"]= './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for word in words:
    print word
    os.mkdir(TARGET_DIRECTORY+word)
    for setType in ['/test/','/train/','/val/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        os.mkdir(TARGET_DIRECTORY+word+setType)
        for files in testSet:
            if files.endswith('.mp4'):
                fielDone+=1
                videoFileName = files
                metaFileName = files.replace('mp4','txt')
                metaFile = open(SOURCE_DIRECTORY + word + setType + metaFileName)
                fileSeconds = float(metaFile.readlines()[4].split()[1])
                fileFrames = math.ceil(fileSeconds*25)
                if fileFrames%2==0:
                    fileFrames+=1
                frameRangeStart = int(15- ((fileFrames-1)/2)-1)
                frameRangeEnd = int(15 + ((fileFrames-1)/2)-1)
                frameRangeLength = frameRangeEnd-frameRangeStart+1
                # print '=------------------------Start processing================----------'
                cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
                out = open(TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','txt'),'w+')
                ctr = 0
                printedFrames = 0
                enRects = 0
                temp0=[]
                while(ctr<29):
                    ret, frame = cap.read()
                    if(frameRangeStart<=ctr and ctr<=frameRangeEnd):
                        image = frame
                    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    	rects = detector(gray, 1)
                    	for (i, rect) in enumerate(rects):
                            enRects+=1
                            k = 0
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)
                            for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
                            	# out.write(str(shape[i:j]))
                                temp1=[]
                            	for x, y in shape[i:j]:
                                    temp2=[]
                                    # print x,y
                                    temp2.append(x)
                                    temp2.append(y)
                                    temp1.append(temp2)
                                # print len(temp1)
                                printedFrames+=1
                                break
                        temp0.append(temp1)
                    ctr+=1
                out.write(str(temp0))
                if frameRangeLength!=printedFrames:
                    print 'Not proper'
                    os.remove(TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','txt'))
                    continue
print "=====made files============="
