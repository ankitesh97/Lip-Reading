#################################################
#This file is to load the 24 features file wise
#that is if batch size is 20 the getNextBatch would return 20x29x24 where 29 is the fixed number of frames, 0's are added in the end
#################################################
import math
import os
from Queue import Queue
import numpy as np
import cv2
import json

totalTrainFileName = Queue()
word_to_index = {'ABOUT':0,'BANKS':1,'CONSERVATIVE':2,'DIFFERENCE':3,'ENERGY':4,'FAMILY':5,'GEORGE':6,'HAPPEN':7,'INDEPENDENT':8}

def angle_between(p1,p2):
    if p2[0]-p1[0]!=0:
        slope = float(p2[1]-p1[1])/(p2[0]-p1[0])
        return math.degrees(math.atan(slope))
    else:
        return 90

def distance_between(p1,p2):
    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist


def loadDataQueue(COLORFLAG=0):
    DIRECTORY = '/home/dharin/Desktop/FYP/clustering/lip-border/'
    fileArray = []
    for word in os.listdir(DIRECTORY):
        for fileName in os.listdir(DIRECTORY+word+'/test/'):
            fileArray.append(DIRECTORY+word+'/test/'+fileName)
    np.random.shuffle(fileArray)
    for x in fileArray:
        totalTrainFileName.put(x)
    return len(fileArray)

def emptyDataQueue():
    for x in range(0,totalTrainFileName.qsize()):
        temp = totalTrainFileName.get()

def getNextBatch(batchSize,COLORFLAG=0):
    finalDataReturn = []
    finalNameReturn=[]
    frameReturn=[]
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for counter in range(0,forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        finalNameReturn.append(word_to_index[removedFromQueue.split('_')[0].split('/')[-1]])
        frameReturn.append(int(removedFromQueue.split('_')[2].split('.')[0]))
        featuresArray = np.array(eval(open(removedFromQueue,'r').read()))
        noOfFrames = featuresArray.shape[0]
        framesToAdd = 29-noOfFrames
        features29x24=[]
        for x in range(0,featuresArray.shape[0]):
            features24 = []
            for y in range(0,featuresArray.shape[1]-1):
                current = featuresArray[x]
                anglePt = angle_between(current[y],current[y+1])
                distancePt = distance_between(current[y],current[y+1])
                features24.append(anglePt)
                features24.append(distancePt)
            anglePt = angle_between(current[-1],current[0])
            distancePt = distance_between(current[-1],current[0])
            features24.append(anglePt)
            features24.append(distancePt)
            features29x24.append(features24)
        features29x24 = np.array(features29x24)
        framesToAdd = np.zeros([framesToAdd,24])
        finalData = np.vstack((features29x24,framesToAdd))
        finalDataReturn.append(finalData)
    return np.array(finalDataReturn),np.array(finalNameReturn),np.array(frameReturn)

total = loadDataQueue()
a,b,c = getNextBatch(105)
print a.shape,b.shape,c.shape
