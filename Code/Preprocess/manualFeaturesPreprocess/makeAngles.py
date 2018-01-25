#################################################
#This file is for creating the allTogetherData.txt from the 9wordsX500videos(12 coordinates)
#################################################
import numpy as np
import cv2
import math
import os
from imutils import face_utils
import dlib
from collections import OrderedDict

def angle_between(p1,p2):
    if p2[0]-p1[0]!=0:
        slope = float(p2[1]-p1[1])/(p2[0]-p1[0])
        return math.degrees(math.atan(slope))
    else:
        return 90

def distance_between(p1,p2):
    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

SOURCE_DIRECTORY = './clustering/'
# out = open('smallFeatures.txt','w+')
words = os.listdir(SOURCE_DIRECTORY)
allTogetherData = []
for word in words:
    print word
    for setType in ['/train/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        for files in testSet:
            print SOURCE_DIRECTORY + word + setType + files
            featuresFile = open(SOURCE_DIRECTORY + word + setType + files)
            featuresArray = np.array(eval(featuresFile.read()))
            # print featuresArray.shape
            ############### Conversion to feature starts ##################
            for x in range(0,featuresArray.shape[0]):
                features24 = []
                current = featuresArray[x]
                for y in range(0,len(current)-1):
                    anglePt = angle_between(current[y],current[y+1])
                    distancePt = distance_between(current[y],current[y+1])
                    features24.append(anglePt)
                    features24.append(distancePt)
                anglePt = angle_between(current[-1],current[0])
                distancePt = distance_between(current[-1],current[0])
                features24.append(anglePt)
                features24.append(distancePt)
                allTogetherData.append(features24)
allTogetherDataArray = np.array(allTogetherData)
np.random.shuffle(allTogetherDataArray)
print allTogetherDataArray.shape
allTogetherDataList = allTogetherDataArray.tolist()
out.write(str(allTogetherDataList))
