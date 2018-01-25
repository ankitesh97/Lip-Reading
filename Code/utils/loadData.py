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
def loadDataQueue(COLORFLAG=0,is_val='train'):
    SOURCE_DIRECTORY = '/home/ankitesh/Lip-Reading/Data/lip-seq3/'
    COLOR_SOURCE_DIRECTORY = '/home/ankitesh/Lip-Reading/Data/lip-border-seq/'
    DIRECTORY=''
    if COLORFLAG==0:
        DIRECTORY = SOURCE_DIRECTORY
    else :
        DIRECTORY = COLOR_SOURCE_DIRECTORY
    fileArray = []
    filenames = []
    for word in os.listdir(DIRECTORY):
        for fileName in os.listdir(DIRECTORY+word+'/'+is_val+'/'):
            if 'avi' in fileName:
                fileArray.append(DIRECTORY+word+'/'+is_val+'/'+fileName)
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


def getNextBatch(batchSize,COLORFLAG=0,is_train='train'):
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
        manual_features.append(getFeatureVector(fileNameFromQueue,seq_len,word,is_train))
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
    return np.array(np.swapaxes(finalDataReturn,0,1)),np.array(finalNameReturn), np.array(frameReturn), np.swapaxes(np.array(manual_features),0,1)
# loadDataQueue()
# a,b,c = getNextBatch(5)
# print b,c

def getFeatureVector(filename,seq_len,word,is_train):
    BASE_ADD = '/home/ankitesh/Lip-Reading/Data/lip-border3/'
    removedFromQueue = filename.split('.avi')[0]+'.txt'
    removedFromQueue = BASE_ADD+word+'/'+is_train+'/'+removedFromQueue
    featuresArray = np.array(eval(open(removedFromQueue,'r').read()))
    noOfFrames = featuresArray.shape[0]
    framesToAdd = 29-noOfFrames
    features29x24=[]
    for x in range(0,seq_len):
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

    finalDataReturn = []
    features29x24 = np.array(features29x24)
    framesToAdd = np.zeros([framesToAdd,24])
    finalData = np.vstack((features29x24,framesToAdd))
    finalDataReturn.append(finalData)
    return np.array(finalDataReturn).reshape(29,24) # time_seq x 24
