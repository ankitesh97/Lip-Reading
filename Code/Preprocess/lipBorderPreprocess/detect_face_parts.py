#################################################
#This file is to give vidofile which has only lip border
#using the facial landmarks, this is for an individual video only, for testing purpose only
#################################################
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
from collections import OrderedDict

def getLipBorder(readFileName):
	cap = cv2.VideoCapture(readFileName)
	args={}
	FACIAL_LANDMARKS_IDXS = OrderedDict([
		("mouth", (48, 60))
	])
	args["shape_predictor"]= './shape_predictor_68_face_landmarks.dat'
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc,1,(72, 43),isColor=0)
	ctr = 0
	while(ctr<29):
		ctr+=1
		emptyImage = np.zeros((256,256,3),np.uint8)
		ret,args["image"]= cap.read()
		image = args["image"]
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 1)
		for (i, rect) in enumerate(rects):
			k = 0
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
			for (name, (i, j)) in FACIAL_LANDMARKS_IDXS.items():
				clone = image.copy()
				pv_x,pv_y = shape[i][0],shape[i][1]
				first_x,first_y =pv_x,pv_y
				for (x, y) in shape[i+1:j]:
					cv2.circle(clone, (x, y), 1, (0, 255, 0), -1)
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
				frameWritte+=1
				k = cv2.waitKey(30) & 0xff
				break

getLipBorder('./ABOUT_00001.mp4')
