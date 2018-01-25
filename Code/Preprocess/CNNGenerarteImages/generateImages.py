#################################################
#This file is to generate lip and nonlip images from the original data and store it in two separate folders
#################################################
import os
import math
import cv2
import numpy as np
imageCounter=0
MAX_IMAGE_COUNTER = 3670*4
COLORFLAG = 0
mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
SOURCE_DIRECTORY = '../Lip_Reading_Data/lipread_mp4/'
LIP_DIRECTORY = 'lip/'
NONLIP_DIRECTORY = 'nonlip/'
COLOR_LIP_DIRECTORY = 'lip-color/'
COLOR_NONLIP_DIRECTORY = 'nonlip-color/'
L_DIR = ''
NL_DIR = ''
if COLORFLAG==0:
    L_DIR = LIP_DIRECTORY
    NL_DIR = NONLIP_DIRECTORY
else:
    L_DIR = COLOR_LIP_DIRECTORY
    NL_DIR = COLOR_NONLIP_DIRECTORY

setType = '/val/'
words = os.listdir(SOURCE_DIRECTORY)
numberOfWords = len(words)
videoPerWord = int(MAX_IMAGE_COUNTER/numberOfWords)
print videoPerWord
finalWrittenCount = 0
for word in words:
    print word
    tempTestSet = os.listdir(SOURCE_DIRECTORY + word + setType)
    testSet = []
    # np.random.shuffle(testSet)
    for x in tempTestSet:
        if 'mp4' in x:
            testSet.append(x)
    np.random.shuffle(testSet)
    # print testSet
    wordFileName = []
    fileNumber=0
    fileAccNumber = 0
    for fileAccNumber in range(0,len(testSet)):
        if fileNumber==videoPerWord:
            break
        # fileAccNumber+=1
        videoFileName = testSet[fileAccNumber]
        cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
        ctr = 0
        while(ctr<29):
            ret,frame = cap.read()
            tempVar = frame.shape[0]
            extraFrame = frame[:int(tempVar/2.0),:]
            lipFrame = frame[int(tempVar/2.0):tempVar,:]
            gray = cv2.cvtColor(lipFrame, cv2.COLOR_BGR2GRAY)
            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 5)
            if len(mouth_rects)!=0:
                ourSizeFound = False
                for x in range(0,len(mouth_rects)):
                    if mouth_rects[x][2]==43 and mouth_rects[x][3]==72:
                        ourSizeFound = True
                        x,y,w,h = mouth_rects[x][0],mouth_rects[x][1],mouth_rects[x][2],mouth_rects[x][3]
                        y = max(int(y - 0.15*h),0)
                        break
                if ourSizeFound == True:
                    fileNumber+=1
                    cropped_lip = lipFrame[y:y+w,x:x+h,0]
                    # print extraFrame.shape
                    cv2.imwrite(L_DIR+"LIP_"+str(finalWrittenCount)+".jpg",cropped_lip)
                    randWidth = np.random.randint(256)
                    randHeight = np.random.randint(128)
                    sW,eW,sH,eH = 0,0,0,0
                    if randWidth+72>255:
                        eW = randWidth
                        sW = randWidth-72
                    else:
                        sW = randWidth
                        eW = randWidth+72
                    if randHeight+43>127:
                        eH = randHeight
                        sH = randHeight-43
                    else:
                        sH = randHeight
                        eH = randHeight+43
                    cropped_nonlip = extraFrame[sH:eH,sW:eW,0]
                    cv2.imwrite(NL_DIR+"NONLIP_"+str(finalWrittenCount)+".jpg",cropped_nonlip)
                    finalWrittenCount+=1
                    break
            ctr+=1
    print fileNumber
        # print SOURCE_DIRECTORY+word+setType+testSet[fileAccNumber]
