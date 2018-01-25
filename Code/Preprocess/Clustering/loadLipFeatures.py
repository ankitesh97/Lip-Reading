#################################################
#This file is used by clustering.py for loading all jumbled up frames 24 features from allTogetherDataFeatures.txt
#and to get the batch so the clustering model can be trained
#################################################
import os
from Queue import Queue
import numpy as np
import cv2
import json


totalTrainFileName = Queue()
def loadDataQueue(COLORFLAG=0):
    L_DIRECTORY = '/home/dharin/Desktop/FYP/clustering/allTogetherDataFeatures.txt'
    with open(L_DIRECTORY) as inp:
        fileList = eval(inp.read())
    for x in fileList:
        totalTrainFileName.put(x)
    return len(fileList)

def emptyDataQueue():
    for x in range(0,totalTrainFileName.qsize()):
        temp = totalTrainFileName.get()

def getNextBatch(batchSize,COLORFLAG=0):
    finalDataReturn = []
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for x in range(forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        finalDataReturn.append(removedFromQueue)
    finalDataReturn = np.array(finalDataReturn)
    return finalDataReturn
