#################################################
#This file helps in loading data batch wise during training of the model1,
#getNextBatch only returns data,tag not frame size in this file so specially for model1
#################################################

import os
from Queue import Queue
import numpy as np
import cv2
from wordDict import *

totalTrainFileName = Queue()

def loadDataQueue(COLORFLAG=0):
    SOURCE_DIRECTORY = './modified/'
    COLOR_SOURCE_DIRECTORY = './modified-color/'
    DIRECTORY=''
    if COLORFLAG==0:
        DIRECTORY = SOURCE_DIRECTORY
    else :
        DIRECTORY = COLOR_SOURCE_DIRECTORY
    fileArray = []
    # if COLORFLAG==0:
    for word in os.listdir(DIRECTORY):
        for fileName in os.listdir(DIRECTORY+word+'/test/'):
            if 'mp4' in fileName:
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
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for x in range(forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        # print removedFromQueue
        finalNameReturn.append(word_to_index[removedFromQueue.split('_')[0].split('/')[-1]])
        cap = cv2.VideoCapture(removedFromQueue)
        temp=[]
        for x in range(0,29):
            ret,frame = cap.read()
            frame = np.swapaxes(frame, 0,1)
            if COLORFLAG==0:
                temp.append(frame[:,:,0])
            else:
                temp.append(frame[:,:,:])
        if COLORFLAG==0:
            temp2 = np.array(temp)
            temp3 = temp2.reshape(temp2.shape+(1,))
        else:
            temp3 = temp
        finalDataReturn.append(temp3)
    return np.array(np.swapaxes(finalDataReturn,0,1)),np.array(finalNameReturn)
