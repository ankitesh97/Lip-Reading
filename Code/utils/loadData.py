import os
from Queue import Queue
import numpy as np
import cv2

COLORFLAG = 0
SOURCE_DIRECTORY = './modified/'
COLOR_SOURCE_DIRECTORY = './modified-color/'
DIRECTORY=''
if COLORFLAG==0:
    DIRECTORY = SOURCE_DIRECTORY
else :
    DIRECTORY = COLOR_SOURCE_DIRECTORY
totalTrainFileName = Queue()

def loadDataQueue():
    fileArray = []
    # if COLORFLAG==0:
    for word in os.listdir(DIRECTORY):
        for fileName in os.listdir(DIRECTORY+word+'/test/'):
            if 'mp4' in fileName:
                fileArray.append(DIRECTORY+word+'/test/'+fileName)
    np.random.shuffle(fileArray)
    for x in fileArray:
        totalTrainFileName.put(x)


def getNextBatch(batchSize):
    finalDataReturn = []
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for x in range(forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        # print removedFromQueue
        cap = cv2.VideoCapture(removedFromQueue)
        temp=[]
        for x in range(0,29):
            ret,frame = cap.read()
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
    return np.array(finalDataReturn)
#
# loadDataQueue()
# data = getNextBatch(4)
# print data
# data = getNextBatch(4)
# print data.shape
# data = getNextBatch(4)
# print data.shape
# data = getNextBatch(4)
# print data.shape
#
# print totalTrainFileName.qsize()
