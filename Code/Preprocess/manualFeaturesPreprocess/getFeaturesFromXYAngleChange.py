#################################################
#This file is for getting framesx24 features in separate txt files which change orientation of angles
#the directory is mentioned in target and needs coordinates as source
#################################################

import numpy as np
import cv2
import math
import os
from imutils import face_utils
import dlib
from collections import OrderedDict

def convertToNormal(current):
    # print np.array(current).shape
    newList =[]
    for x in range(0,len(current)):
        newCoOd = []
        newCoOd.append(current[x][0])
        newCoOd.append(256-current[x][1])
        newList.append(newCoOd)
    # print np.array(newList).shape
    return newList
def angle_between(baseSlope,p1,p2):
    if p2[0]-p1[0]!=0:
        slope = float(p2[1]-p1[1])/(p2[0]-p1[0])
        tnAngle = slope-baseSlope/1+(baseSlope*slope)
        return math.degrees(math.atan(tnAngle))
    else:
        return 90

def distance_between(p1,p2):
    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

SOURCE_DIRECTORY = './9wordsX1000videosXY/'
TARGET_DIRECTORY = './modified-features/'

# out = open('smallFeatures.txt','w+')
words = os.listdir(SOURCE_DIRECTORY)
for word in words:
    print word
    os.mkdir(TARGET_DIRECTORY+word)
    for setType in ['/test/','/train/','/val/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        os.mkdir(TARGET_DIRECTORY+word+setType)
        for files in testSet:
            out = open(TARGET_DIRECTORY + word + setType + files,'w+')
            allTogetherData = []
            # print SOURCE_DIRECTORY + word + setType + files
            featuresFile = open(SOURCE_DIRECTORY + word + setType + files)
            featuresArray = np.array(eval(featuresFile.read()))
            # print featuresArray.shape
            ############### Conversion to feature starts ##################
            for x in range(0,featuresArray.shape[0]):
                features24 = []
                current = featuresArray[x]
                # print np.array(current).shape
                current = convertToNormal(current)
                # print np.array(current).shape
                firstPoint = current[0]
                seventhPoint = current[6]
                # print firstPoint,seventhPoint
                if seventhPoint[0]-firstPoint[0]!=0:
                    baseSlope = float(seventhPoint[1]-firstPoint[1])/(seventhPoint[0]-firstPoint[0])
                else:
                    print 'gadbad hai bhai'
                    baseSlope = 0
                # print baseSlope
                for y in range(0,len(current)-1):
                    # print current[y],current[y+1]
                    anglePt = angle_between(baseSlope,current[y],current[y+1])
                    # print anglePt
                    distancePt = distance_between(current[y],current[y+1])
                    features24.append(anglePt)
                    features24.append(distancePt)
                anglePt = angle_between(baseSlope,current[-1],current[0])
                distancePt = distance_between(current[-1],current[0])
                features24.append(anglePt)
                features24.append(distancePt)
                allTogetherData.append(features24)
            # print np.array(allTogetherData).shape
            out.write(str(allTogetherData))
            # break
    # break
# allTogetherDataArray = np.array(allTogetherData)
# np.random.shuffle(allTogetherDataArray)
# print allTogetherDataArray.shape
# allTogetherDataList = allTogetherDataArray.tolist()
# out.write(str(allTogetherDataList))
