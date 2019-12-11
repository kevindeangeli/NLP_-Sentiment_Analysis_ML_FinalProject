'''
Created by Kevin De Angeli
Date: 2019-12-07
'''

import copy
import numpy as np
import time #For computing time


class Kmeans:
    def __init__(self, k=2):
        self.k=k
        self.C = []
        self.iterations = 0
        self.X = []
        self.ClusterClass = np.array([-1,-1])
        self.Y =[]


    def fit(self, x_train, y_train, iterationsLimit=-1):
        timeStart = time.time()

        self.k = np.unique(y_train).shape[0]
        vectorDimensions = x_train[0].shape[0]

        self.C.append(x_train[7])
        self.C.append(x_train[7])
        #self.C = [np.random.randint(low=0, high=1, size=vectorDimensions) for i in range(self.k)]
        self.X = x_train
        self.C = np.array(self.C)
        self.Y = y_train


        C_old = np.array([])
        while not self.finishLoopCheck(oldClusters=C_old, iterationsLim=iterationsLimit):
            print("Iteration: ", self.iterations)
            C_old = copy.deepcopy(self.C)  # To copy C by value not by reference
            dataAssignment = self.closestCluster()
            self.clustersUpdate(dataAssignment)
            self.iterations += 1

        if iterationsLimit == self.iterations:
            clusterAssignment = []
            for i in self.X:  # For each dataPoint
                dist = []
                for k in self.C:  # For each cluster.
                    dist.append(np.linalg.norm(i - k))
                min = np.amin(dist)
                index = dist.index(min)
                clusterAssignment.append(index)

            clusterAssignment=np.array(clusterAssignment)
            ClusterOne = clusterAssignment == 1
            countPositives = np.sum(self.Y[ClusterOne])
            print(countPositives)
            if countPositives >= (clusterAssignment.shape[0]/2):
                self.ClusterClass[0]=1
                self.ClusterClass[1]=0
            else:
                self.ClusterClass[0]=0
                self.ClusterClass[1]=1
        print("--- %s seconds ---" % (time.time() - timeStart))

    def predict(self, y_test):
        self.X= y_test

        clusterAssigned = self.closestCluster()
        classifyAccordingly = clusterAssigned == 0
        predictions = copy.deepcopy(clusterAssigned)  # To copy C by value not by reference
        predictions[classifyAccordingly] = self.ClusterClass[0]
        classifyAccordingly = clusterAssigned == 1
        predictions[classifyAccordingly] = self.ClusterClass[1]
        return predictions





    def reInitializeEmptyClusters(self, CIndex):
        '''
        Re-initialize clusters at randon.
        This is used when clusters are empty.
        '''

        newCoordinates = np.random.randint(low=0, high=256, size=self.X.shape[1])
        self.C[CIndex] = np.array(newCoordinates)

    def clustersUpdate(self, clusterAssignments):
        '''
        In order to handle "empty clusters" I re-initialized those clusters randonly.
        '''
        # clusterAssignments = np.array(clusterAssignments)
        newClusterCoordinate = []
        # update self.C based on clusterAssignments

        for i in range(self.C.shape[0]):
            if i not in clusterAssignments:
                print("Empty Cluster: ", i)
                self.reInitializeEmptyClusters(CIndex=i)
                continue
            findDataPoints = clusterAssignments == i

            dataPointsCoordinates = self.X[findDataPoints]
            newClusterCoordinate = np.average(dataPointsCoordinates, axis=0)
            self.C[i] = newClusterCoordinate



    def finishLoopCheck(self, oldClusters, iterationsLim):
        '''
        Stop the program if the clusters' position stop changing or
        the limit number of iterations has been reached.
        '''
        if iterationsLim == self.iterations:
            return True
        else:
            return np.array_equal(oldClusters, self.C)  # Clusters didn't change ?

    def closestCluster(self):
        '''
        Create a list where each data point is associated with a
        clusters. Then it returns the list of clusters.


        '''
        clusterAssignment = []
        for i in self.X:  # For each dataPoint
            dist = []
            for k in self.C:  # For each cluster.
                dist.append(np.linalg.norm(i - k))
            min = np.amin(dist)
            index = dist.index(min)
            clusterAssignment.append(index)

        # return a list of size X where each element specifies the cluster.
        return np.array(clusterAssignment)
