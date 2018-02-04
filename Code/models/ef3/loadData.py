# def load_src(name, fpath):
#     import os, imp
#     return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))
import os
from Queue import Queue
import numpy as np
import cv2
import math
# load_src("wordDict", "./wordDict.py")
# from wordDict import *
word_to_index = {'ABOUT':0,'BANKS':1,'CONSERVATIVE':2,'DIFFERENCE':3,'ENERGY':4,'FAMILY':5,'GEORGE':6,'HAPPEN':7,'INDEPENDENT':8}
totalTrainFileName = Queue()
filenameQ = Queue()
def loadDataQueue(COLORFLAG=0,data='train'):
    SOURCE_DIRECTORY = '../../../Data/lip-seq/'
    COLOR_SOURCE_DIRECTORY = '../../../Data/lip-seq/'
    DIRECTORY=''
    if COLORFLAG==0:
        DIRECTORY = SOURCE_DIRECTORY
    else :
        DIRECTORY = COLOR_SOURCE_DIRECTORY
    fileArray = []
    filenames = []
    for word in os.listdir(DIRECTORY):
        for fileName in os.listdir(DIRECTORY+word+'/'+data+'/'):
            if 'avi' in fileName:
                fileArray.append(DIRECTORY+word+'/'+data+'/'+fileName)
                filenames.append(fileName)
    zipped = zip(fileArray,filenames)
    np.random.shuffle(zipped)
    fileArray, filenames = zip(*zipped)
    for i in range(len(fileArray)):
        totalTrainFileName.put(fileArray[i])
        filenameQ.put(filenames[i])
    return len(fileArray)

def emptyDataQueue():
    for x in range(0,totalTrainFileName.qsize()):
        temp = totalTrainFileName.get()
        temp2 = filenameQ.get()


def angle_between(p1,p2):
    if p2[0]-p1[0]!=0:
        slope = float(p2[1]-p1[1])/(p2[0]-p1[0])
        return math.degrees(math.atan(slope))
    else:
        return 90

def distance_between(p1,p2):
    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist


def getNextBatch(batchSize,COLORFLAG=0,data='train'):
    finalDataReturn = []
    finalNameReturn=[]
    frameReturn=[]
    manual_features = []
    if batchSize<=totalTrainFileName.qsize():
        forLoopRange = batchSize
    else:
        forLoopRange = totalTrainFileName.qsize()
    for x in range(forLoopRange):
        removedFromQueue = totalTrainFileName.get()
        fileNameFromQueue = filenameQ.get()
        finalNameReturn.append(word_to_index[removedFromQueue.split('_')[0].split('/')[-1]])
        frameReturn.append(int(removedFromQueue.split('_')[2].split('.')[0]))
        seq_len = frameReturn[-1]
        word = removedFromQueue.split('_')[0].split('/')[-1]
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
        else:
            temp3 = temp

        finalDataReturn.append(temp2.reshape(72,42,29))
    finalDataReturn = np.array(finalDataReturn)
    return finalDataReturn,np.array(finalNameReturn), np.array(frameReturn)
# loadDataQueue()
# a,b,c = getNextBatch(5)
# print a.shape
