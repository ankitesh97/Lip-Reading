
import os
from Queue import Queue
import numpy as np
import cv2
import json

totalTrainFileName = Queue()
def loadDataQueue(COLORFLAG=0):
    L_DIRECTORY = '/home/ankitesh/Lip-Reading/Data/features'
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
        # print f.read()
        features = json.loads(f.read())["feature"]
        finalDataReturn.append(features)

    finalDataReturn = np.array(finalDataReturn)
    print finalDataReturn.shape
    return finalDataReturn

total = loadDataQueue()

a= getNextBatch(total)
