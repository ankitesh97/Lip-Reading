#################################################
#This file was to simultaneously load all coordinates make angles
#give it as input to the kmeans algorithm and then after pca plot some data to visualize it.
#################################################

from sklearn import cluster
import numpy as np
import pickle
from loadLipFeatures import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import math
import os
from imutils import face_utils
import dlib
from collections import OrderedDict



class KMEANS:
    def __init__(self,address,noOfCluster=20,max_iter=1000,n_init=50):
        self.noOfCluster = noOfCluster
        self.max_iter = max_iter
        self.n_init = n_init
        self.address = address
        # with open(address, 'rb') as input:
        #     kmeans = pickle.load(input)
        self.kmeans = None

    def train(self,data):
        kmeans = cluster.KMeans(n_clusters=self.noOfCluster,max_iter=self.max_iter,n_init = self.n_init)
        kmeans.fit(data)
        self.kmeans = kmeans
        with open(self.address, 'wb') as output:
            pickle.dump(kmeans,output)

    def loadFile(self):
        with open(self.address, 'rb') as input:
            kmeans = pickle.load(input)
            self.kmeans = kmeans

    def predict(self,data):
        predicted = self.kmeans.predict(data)
        y2 = predicted.reshape(-1)
        one_hot_targets = np.eye(self.noOfCluster)[y2]
        return one_hot_targets

    def getLabels(self):
        return self.kmeans.labels_

    def trainForElbow(self,data,n_jobs=-1):
        kmeans = cluster.KMeans(n_clusters=self.noOfCluster,max_iter=self.max_iter,n_init = self.n_init,n_jobs=-1)
        kmeans.fit(data)
        self.kmeans=kmeans
        with open(self.address, 'wb') as output:
            pickle.dump(kmeans,output)


def angle_between(p1,p2):
    if p2[0]-p1[0]!=0:
        slope = float(p2[1]-p1[1])/(p2[0]-p1[0])
        return math.degrees(math.atan(slope))
    else:
        return 90

def distance_between(p1,p2):
    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    return dist

SOURCE_DIRECTORY = './clustering/'
# out = open('smallFeatures.txt','w+')
words = os.listdir(SOURCE_DIRECTORY)
allTogetherData = []
for word in words:
    print word
    for setType in ['/train/']:
        testSet = os.listdir(SOURCE_DIRECTORY + word + setType)
        for files in testSet:
            print SOURCE_DIRECTORY + word + setType + files
            featuresFile = open(SOURCE_DIRECTORY + word + setType + files)
            featuresArray = np.array(eval(featuresFile.read()))
            # print featuresArray.shape
            ############### Conversion to feature starts ##################
            for x in range(0,featuresArray.shape[0]):
                features24 = []
                current = featuresArray[x]
                for y in range(0,len(current)-1):
                    anglePt = angle_between(current[y],current[y+1])
                    distancePt = distance_between(current[y],current[y+1])
                    features24.append(anglePt)
                    features24.append(distancePt)
                anglePt = angle_between(current[-1],current[0])
                distancePt = distance_between(current[-1],current[0])
                features24.append(anglePt)
                features24.append(distancePt)
                allTogetherData.append(features24)
            break
allTogetherDataArray = np.array(allTogetherData)
print allTogetherDataArray.shape
data = allTogetherDataArray
noOfCluster = 20
obj = KMEANS('./kmeans-20.pkl')
obj.loadFile()
predictions = obj.predict(data)
print predictions
# labels = obj.getLabels()
# print len(labels)
labels = np.argmax(predictions,axis=1)
print labels
nf = 2
pca = PCA(n_components = nf)
pca.fit(data)
data_new = pca.transform(data)
print data_new.shape
clusterToColor={}
addPerCluster = 1.0/noOfCluster
startValue = 0.0
for x in range(0,noOfCluster):
    clusterToColor[x]=tuple([0.0+x*addPerCluster,1.0-x*addPerCluster,0.0+x*addPerCluster])
for x in range(0,data.shape[0]):
    plt.scatter(data_new[x][0],data_new[x][1],c=clusterToColor[labels[x]])
plt.show()
