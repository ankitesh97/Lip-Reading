#################################################
#This file is to generate lip images for training CNN but only border using facial landmarks
#################################################
import os
import math
import cv2
import numpy as np
from imutils import face_utils
import dlib
from collections import OrderedDict

imageCounter=0
MAX_IMAGE_COUNTER = 3670*4
COLORFLAG = 0
SOURCE_DIRECTORY = '../Lip_Reading_Data/lipread_mp4/'
LIP_DIRECTORY = 'lipBorder/'
NONLIP_DIRECTORY = 'nonlip/'
L_DIR = LIP_DIRECTORY
NL_DIR = NONLIP_DIRECTORY
args={}
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 60))
])
args["shape_predictor"]= './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fourcc = cv2.VideoWriter_fourcc(*'XVID')


setType = '/train/'
words = os.listdir(SOURCE_DIRECTORY)
numberOfWords = len(words)
videoPerWord = int(MAX_IMAGE_COUNTER/numberOfWords)
print videoPerWord
finalWrittenCount = 0
for word in words:
    print words.index(word),word
    tempTestSet = os.listdir(SOURCE_DIRECTORY + word + setType)
    testSet = []
    # np.random.shuffle(testSet)
    for x in tempTestSet:
        if 'mp4' in x:
            testSet.append(x)
    np.random.shuffle(testSet)
    # print testSet
    wordFileName = []
    fileNumber=0
    fileAccNumber = 0
    for fileAccNumber in range(0,len(testSet)):
        if fileNumber==videoPerWord:
            break
        videoFileName = testSet[fileAccNumber]
        cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
        ctr = 0
        while(ctr<1):
            ret,frame = cap.read()
            emptyImage = np.zeros((256,256,3),np.uint8)
            image = frame
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):
                # enRects+=1
                k = 0
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
                    clone = image.copy()
                    pv_x,pv_y = shape[i][0],shape[i][1]
                    first_x,first_y =pv_x,pv_y
                    for (x, y) in shape[i+1:j]:
                    	start = (pv_x,pv_y)
                    	end = (x,y)
                    	cv2.line(emptyImage,start,end,[255,255,255],1)
                    	pv_x,pv_y = x,y
                    cv2.line(emptyImage,(pv_x,pv_y),(first_x,first_y),[255,255,255],1)
                    (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                    paddingWL = int(math.floor((72-w)/2.0))
                    paddingWR = int(math.ceil((72-w)/2.0))
                    paddingHU = int(math.floor((43-h)/2.0))
                    paddingHD = int(math.ceil((43-h)/2.0))
                    roi = emptyImage[y-paddingHU:y + h+paddingHD, x-paddingWL:x + w+paddingWR]
                    temp = np.array(shape[i:j])
                    cnt = np.reshape(temp,(12,1,2))
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(L_DIR+"LIP_"+str(finalWrittenCount)+".jpg",roi)
                    finalWrittenCount+=1
                    fileNumber+=1
                    # printedFrames+=1
                    break
            ctr+=1
