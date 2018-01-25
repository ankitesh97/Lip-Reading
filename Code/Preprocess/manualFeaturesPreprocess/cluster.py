#################################################
#This file is for the kmeans class and the below commented code is for loading data
#and training the kmeans object and also plotting the results
#################################################
from sklearn import cluster
import numpy as np
import pickle
from loadLipFeatures import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class KMEANS:
    def __init__(self,address,noOfCluster=17,max_iter=1000,n_init=50):
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
        with open('kmeans.pkl', 'wb') as output:
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

######## Data fetching ###############
total = loadDataQueue()
dataSize = total
data = getNextBatch(total)
print data.shape

obj = KMEANS('./kmeans.pkl')
# obj.loadFile()
# predictions = obj.predict(data)
# print predictions
#### Plotting #######################
# obj = KMEANS()
obj.train(data)
# obj.loadFile()
labels = obj.getLabels()
print len(labels)
noOfCluster = 17
nf = 2
pca = PCA(n_components = nf)
pca.fit(data)
data_new = pca.transform(data[:1000])
print data_new.shape
clusterToColor={}
addPerCluster = 1.0/noOfCluster
startValue = 0.0
for x in range(0,noOfCluster):
    clusterToColor[x]=tuple([0.0+x*addPerCluster,1.0-x*addPerCluster,0.0+x*addPerCluster])
for x in range(0,1000):
    plt.scatter(data_new[x][0],data_new[x][1],c=clusterToColor[labels[x]])
plt.show()
