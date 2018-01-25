#################################################
#This file is to create preprocessed data from the original data
#using opencv haar cascade which is got from haarcascade_mcs_mouth.xml in this directory
#this code includes adding frame number to the file name during preprocessing
#################################################
import os
import math
import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
if mouth_cascade.empty():
  raise IOError('Unable to load the mouth cascade classifier xml file')
globalCounterPrevious = 0
globalCounterTotal = 0
SOURCE_DIRECTORY = 'original-large/'
TARGET_DIRECTORY = 'modified/'
COLOR_TARGET_DIRECTORY = 'modified-color/'
words = os.listdir(SOURCE_DIRECTORY)
bakwasFile = open('bakwasFile.txt','w')
for word in words:
    print word
    os.mkdir(TARGET_DIRECTORY+word)
    os.mkdir(COLOR_TARGET_DIRECTORY+word)
    for setType in ['/test/','/train/','/val/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        os.mkdir(TARGET_DIRECTORY+word+setType)
        os.mkdir(COLOR_TARGET_DIRECTORY+word+setType)
        for files in testSet:
            if files.endswith('.mp4'):
                videoFileName = files
                metaFileName = files.replace('mp4','txt')
                metaFile = open(SOURCE_DIRECTORY + word + setType + metaFileName)
                fileSeconds = float(metaFile.readlines()[4].split()[1])
                # print math.floor((1.16-fileSeconds)/2.0 *25) , math.ceil((1.16+fileSeconds)/2.0 *25)
                fileFrames = math.ceil(fileSeconds*25)
                if fileFrames%2==0:
                    fileFrames+=1
                frameRangeStart = int(15- ((fileFrames-1)/2)-1)
                frameRangeEnd = int(15 + ((fileFrames-1)/2)-1)
                frameRangeLength = frameRangeEnd-frameRangeStart+1
                stretchFrames={}
                for i in range(0,29):
                    try:
                        stretchFrames[int(round(i*frameRangeLength/29))+frameRangeStart]+=1
                    except:
                        stretchFrames[int(round(i*frameRangeLength/29))+frameRangeStart]=1
                # print stretchFrames
                # print frameRangeStart,frameRangeEnd,videoFileName,setType
                cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','avi'),fourcc,5,(72, 43),isColor=0)
                outColor = cv2.VideoWriter(COLOR_TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','avi'),fourcc,5,(72, 43),isColor=1)
                ctr = 0
                default_mouth_rects = []
                bakwasFileFlag = False
                # print '=======First Loop Starts=============='
                while(ctr<29):
                    ret, frame = cap.read()
                    tempVar = frame.shape[0]
                    frame = frame[int(tempVar/2.0):tempVar,:]
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 5)
                    if len(mouth_rects)!=0:
                        # print mouth_rects
                        finalIndex = 10
                        for x in range(0,len(mouth_rects)):
                            if mouth_rects[x][2]==43 and mouth_rects[x][3]==72:
                                finalIndex = x
                        if finalIndex!=10:
                            default_mouth_rects = [mouth_rects[finalIndex]]
                            break
                    if ctr==28 and len(default_mouth_rects)==0:
                        bakwasFile.write('./'+TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','avi')+'\n')
                        bakwasFile.write('./'+COLOR_TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','avi')+'\n')
                        bakwasFileFlag = True
                    ctr+=1
                # print '=======First Loop Ends=============='
                printedFrames = 0
                if bakwasFileFlag ==True:
                    pass
                    # print 'Omitting file ',videoFileName
                else:
                    ctr = 0
                    # print default_mouth_rects
                    previous_mouth_rects = default_mouth_rects
                    previous_mouth_rects[0][2]=43
                    previous_mouth_rects[0][3]=72
                    cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
                    mouth_rects=[]
                    # print '=======Second Loop Starts=============='
                    while(ctr<29):
                        # print ctr
                        ret, frame = cap.read()
                        tempVar = frame.shape[0]
                        if (frameRangeStart<=ctr and ctr<=frameRangeEnd):
                            frame = frame[int(tempVar/2.0):tempVar,:]
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            if len(mouth_rects)!=0:
                                finalIndex = 10
                                for x in range(0,len(mouth_rects)):
                                    if mouth_rects[x][2]==43 and mouth_rects[x][3]==72:
                                        finalIndex = x
                                if finalIndex!=10:
                                    previous_mouth_rects = [mouth_rects[finalIndex]]
                            mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7,5)
                            globalCounterTotal+=1
                            try:
                                finalIndex = 10
                                for x in range(0,len(mouth_rects)):
                                    if mouth_rects[x][2]==43 and mouth_rects[x][3]==72:
                                        finalIndex = x
                                x,y,w,h = mouth_rects[finalIndex][0],mouth_rects[finalIndex][1],mouth_rects[finalIndex][2],mouth_rects[finalIndex][3]
                                y = max(int(y - 0.15*h),0)
                            except :
                                # print ctr, "Previous Liya",previous_mouth_rects,default_mouth_rects
                                globalCounterPrevious+=1
                                finalIndex = 0
                                for x in range(0,len(previous_mouth_rects)):
                                    if previous_mouth_rects[x][2]==43 and previous_mouth_rects[x][3]==72:
                                        finalIndex = x
                                x,y,w,h = previous_mouth_rects[finalIndex][0],previous_mouth_rects[finalIndex][1],previous_mouth_rects[finalIndex][2],previous_mouth_rects[finalIndex][3]
                            cropped_img = frame[y:y+w,x:x+h]
                            # print cropped_img.shape
                            final_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                            final_img_color = cropped_img
                            #for timesToWrite in range(0,stretchFrames[ctr]):
                            out.write(final_img)
                            outColor.write(final_img_color)
                            #printedFrames+=stretchFrames[ctr]
			    printedFrames+=1
                        ctr+=1
                    # print '=======Second Loop Ends=============='
                    # print printedFrames, fileFrames
                    for paddingFrame in range(0,int(29-fileFrames)):
                         black_image = np.zeros((43,72,3), np.uint8)
                         final_img = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
                         final_img_color = black_image
                         out.write(final_img)
                         outColor.write(final_img_color)
                         printedFrames+=1
                    if printedFrames!=29:
                        print 'Not proper'
bakwasFile.close()
print 'Took Previous frames % :',float(globalCounterPrevious)/globalCounterTotal*100,"%"
f = open('./bakwasFile.txt','r')
lines = f.readlines()
for x in lines:
    print x
    os.remove(x[:-1])
