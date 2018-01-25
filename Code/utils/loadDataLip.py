import os
from Queue import Queue
import numpy as np
import cv2
word_to_index = {"LIP":1,"NONLIP":0}
totalTrainFileName = Queue()
def loadDataQueue(COLORFLAG=0):
    L_DIRECTORY = '/home/dharin/Desktop/Lip-Reading/Data/lip'
    NL_DIRECTORY = '/home/ankitesh/Lip-Reading/Data/nonlip-border'
    fileArray = []
    # if COLORFLAG==0:
    # print os.system("pwd")
    for fileName in os.listdir(L_DIRECTORY):
        fileArray.append(L_DIRECTORY+'/'+fileName)
    # for fileName in os.listdir(NL_DIRECTORY):
        # fileArray.append(NL_DIRECTORY+'/'+fileName)
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
        cap = cv2.imread(removedFromQueue,0)
        cap = cap[1:43,:]

        finalDataReturn.append(cap)


        # cv2.imshow("img",cap)
        # cv2.waitKey(0)
    finalDataReturn = np.array(np.swapaxes(finalDataReturn,1,2))
    return np.reshape(finalDataReturn, finalDataReturn.shape+(1,)),np.array(finalNameReturn)

# loadDataQueue()
# a,b = getNextBatch(1)
# print b
