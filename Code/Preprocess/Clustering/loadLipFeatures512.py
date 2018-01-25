#################################################
#This file is used by clustering.py for loading all jumbled up frames 512 features from features_lip_border_all_words
#and to get the batch so the clustering model can be trained
#################################################
import os
from Queue import Queue
import numpy as np
import cv2
import json

totalTrainFileName = Queue()
def loadDataQueue(COLORFLAG=0):
    L_DIRECTORY = '/home/dharin/Desktop/FYP/clustering/features_lip_border_all_words'
    fileArray = []
    for fileName in os.listdir(L_DIRECTORY):
        fileArray.append(L_DIRECTORY+'/'+fileName)
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
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for x in range(forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        f = open(removedFromQueue, 'r')
        features = json.loads(f.read())["feature"]
        finalDataReturn.append(features)
    finalDataReturn = np.array(finalDataReturn)
    print finalDataReturn.shape
    return finalDataReturn
