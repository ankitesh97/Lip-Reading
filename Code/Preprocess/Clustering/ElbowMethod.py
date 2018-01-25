#################################################
#This file is to get the optimal number of clusters using the elbow method
#################################################
from scikit import KMEANS
from loadLipFeatures import *
import numpy as np
import pickle
import matplotlib.pyplot as plt

# def load_src(name, fpath):
    # import os, imp
    # return imp.load_source(name, os.path.join(os.path.dirname(__file__), fpath))

cluster_sizes=range(8,33,4)
# load_src("loadLipFeatures", "../../utils/loadLipFeatures.py")


def elbowClusterSelection():
    total = loadDataQueue()
    dataSize = total
    data = getNextBatch(dataSize)

    kmeans=[KMEANS("kmeans-"+str(i)+".pkl",noOfCluster=i,max_iter=1000,n_init=50) for i in cluster_sizes]
    cost=[]
    for kmean in kmeans:
        kmean.trainForElbow(data)
        cost.append(kmean.kmeans.inertia_/float(dataSize))
        print("Done "+str(kmean.noOfCluster))
    plt.plot(cluster_sizes,cost)
    plt.show()

if __name__== "__main__":
    elbowClusterSelection()
