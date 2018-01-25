# def load_src(name, fpath):
#     import os, imp
#     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
import os
from Queue import Queue
import numpy as np
import cv2
# load_src("wordDict", "./wordDict.py")
# from wordDict import *
word_to_index = {'ABOUT':0,'BANKS':1,'CONSERVATIVE':2,'DIFFERENCE':3,'ENERGY':4,'FAMILY':5,'GEORGE':6,'HAPPEN':7,'INDEPENDENT':8}
totalTrainFileName = Queue()

def loadDataQueue(COLORFLAG=0):
    SOURCE_DIRECTORY = '/home/dharin/Desktop/Lip-Reading/Data/9wordsX500videosEndBlack/modified/'
    COLOR_SOURCE_DIRECTORY = '/home/ankitesh/Lip-Reading/Data/doesntMatter/'
    DIRECTORY=''
    if COLORFLAG==0:
        DIRECTORY = SOURCE_DIRECTORY
    else :
        DIRECTORY = COLOR_SOURCE_DIRECTORY
    fileArray = []
    for word in os.listdir(DIRECTORY):
        for fileName in os.listdir(DIRECTORY+word+'/train/'):
            if 'avi' in fileName:
                fileArray.append(DIRECTORY+word+'/train/'+fileName)
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
    # frameReturn=[]
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for x in range(forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        # print removedFromQueue
        finalNameReturn.append(word_to_index[removedFromQueue.split('_')[0].split('/')[-1]])
        # frameReturn.append(int(removedFromQueue.split('_')[2].split('.')[0]))
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
    # ,np.array(frameReturn)
# loadDataQueue()
# a,b,c = getNextBatch(5)
# print b,c
