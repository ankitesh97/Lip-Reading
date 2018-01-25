#################################################
#This file is to generate nonlip images for training CNN but only border using facial landmarks
#################################################

import os
import math
import cv2
import numpy as np
from imutils import face_utils
import dlib
from collections import OrderedDict
MAX_IMAGE_COUNTER = 3670*4
NONLIP_DIRECTORY = 'nonlip-border/'
NL_DIR = NONLIP_DIRECTORY
finalWrittenCount=0
for x in range(0,MAX_IMAGE_COUNTER):
    emptyImage = np.zeros((256,256,3),np.uint8)
    randomPointsW = np.random.randint(0,42,13).reshape(13,1)
    randomPointsH = np.random.randint(0,71,13).reshape(13,1)
    randomNumbers = np.hstack((randomPointsH,randomPointsW))
    previousPoint = tuple(randomNumbers[0])
    firstPoint = tuple(randomNumbers[0])
    for point in range(1,len(randomNumbers)):
        cv2.line(emptyImage,previousPoint,tuple(randomNumbers[point]),[255,255,255],1)
        previousPoint = tuple(randomNumbers[point])
    cv2.line(emptyImage,previousPoint,firstPoint,[255,255,255],1)
    print emptyImage.shape
    cv2.imwrite(NL_DIR+"NONLIP_"+str(finalWrittenCount)+".jpg",emptyImage[:43,:72])
    # cv2.imshow('image',emptyImage[:43,:72])
    # cv2.waitKey(0)
    finalWrittenCount+=1
