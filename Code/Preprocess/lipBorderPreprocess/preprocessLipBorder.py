#################################################
#This file is to generate 9wordsX1000video files from the original image.
#this gives the output lip border preprocessed videos along with number of frames in file name
#################################################
import os
import math
import cv2
import numpy as np
from imutils import face_utils
import dlib
from collections import OrderedDict
fielDone = 0
totalFile = 5500
SOURCE_DIRECTORY = 'original-large/'
TARGET_DIRECTORY = 'lip-border/'
words = os.listdir(SOURCE_DIRECTORY)

args={}
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 60))
])
args["shape_predictor"]= './shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
fourcc = cv2.VideoWriter_fourcc(*'XVID')

for word in words:
    print word
    os.mkdir(TARGET_DIRECTORY+word)
    for setType in ['/test/','/train/','/val/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        os.mkdir(TARGET_DIRECTORY+word+setType)
        for files in testSet:
            if files.endswith('.mp4'):
                fielDone+=1
                videoFileName = files
                metaFileName = files.replace('mp4','txt')
                metaFile = open(SOURCE_DIRECTORY + word + setType + metaFileName)
                fileSeconds = float(metaFile.readlines()[4].split()[1])
                fileFrames = math.ceil(fileSeconds*25)
                if fileFrames%2==0:
                    fileFrames+=1
                frameRangeStart = int(15- ((fileFrames-1)/2)-1)
                frameRangeEnd = int(15 + ((fileFrames-1)/2)-1)
                frameRangeLength = frameRangeEnd-frameRangeStart+1
                # print '=------------------------Start processing================----------'
                cap = cv2.VideoCapture(SOURCE_DIRECTORY+word+setType+videoFileName)
                out = cv2.VideoWriter(TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','avi'),fourcc,1,(72, 43),isColor=0)
                ctr = 0
                printedFrames = 0
                enRects = 0
                while(ctr<29):
                    ret, frame = cap.read()
                    if(frameRangeStart<=ctr and ctr<=frameRangeEnd):
                        emptyImage = np.zeros((256,256,3),np.uint8)
                        image = frame
                    	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    	rects = detector(gray, 1)
                    	for (i, rect) in enumerate(rects):
                            enRects+=1
                            k = 0
                            shape = predictor(gray, rect)
                            shape = face_utils.shape_to_np(shape)
                            for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
                            	clone = image.copy()
                            	pv_x,pv_y = shape[i][0],shape[i][1]
                            	first_x,first_y =pv_x,pv_y
                            	for (x, y) in shape[i+1:j]:
                            		# cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
                            		start = (pv_x,pv_y)
                            		end = (x,y)
                            		cv2.line(emptyImage,start,end,[255,255,255],1)
                            		pv_x,pv_y = x,y
                            	cv2.line(emptyImage,(pv_x,pv_y),(first_x,first_y),[255,255,255],1)
                            	(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                            	paddingWL = int(math.floor((72-w)/2.0))
                            	paddingWR = int(math.ceil((72-w)/2.0))
                            	paddingHU = int(math.floor((43-h)/2.0))
                            	paddingHD = int(math.ceil((43-h)/2.0))
                            	roi = emptyImage[y-paddingHU:y + h+paddingHD, x-paddingWL:x + w+paddingWR]
                            	temp = np.array(shape[i:j])
                            	cnt = np.reshape(temp,(12,1,2))
                            	roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                            	out.write(roi)
                                printedFrames+=1
                                break
                    ctr+=1
                # print frameRangeLength, printedFrames , enRects , videoFileName , setType
                if frameRangeLength!=printedFrames:
                    print 'Not proper'
                    os.remove(TARGET_DIRECTORY+word+setType+videoFileName.replace('.','_'+str(frameRangeLength)+'.').replace('mp4','avi'))
                    continue
                for paddingFrame in range(0,int(29-printedFrames)):
                     black_image = np.zeros((43,72,3), np.uint8)
                     final_img = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
                     final_img_color = black_image
                     out.write(final_img)
                     # outColor.write(final_img_color)
                     printedFrames+=1
                print fielDone/float(totalFile)*100,'%'
