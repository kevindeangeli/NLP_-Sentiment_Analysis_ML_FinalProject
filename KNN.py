'''
Created by Kevin De Angeli
Date: 2019-12-03
'''

import numpy as np
import matplotlib.pyplot as plt


class Knn:
    def __init__(self):
        self.nX = []
        self.pX = []
        self.fX = []
        self.predictionArr =[]
        self.totalTime= -1

    def showTime(self):
        print("Time in seconds: ", self.totalTime)

    def fit(self, X, y):
        #self.nX = normalization(X)
        self.nX = X
        self.pX = pca(self.nX)
        self.fX = fld(self.nX,y)
        self.y = y

    #x here is just a point
    def euclidian_dsitanceList(self, x, X):
        # x is the test point, X is the dataset
        # Using Euclidian Distance
        distancesArr = []
        # For each row in the data set:
        dist = -1
        index=0 # used to associate a distance with a y.
        for row in X:
            testPoint = row[0:X.shape[1]]
            dist = np.linalg.norm(testPoint - x)
            labelIndex = self.y[index]
            distanceAndLabel = (dist, labelIndex)
            distancesArr.append(distanceAndLabel)
            index +=1
        return distancesArr


    #x here is just a point
    def guessLabel(self,x, X, k):
        label0 = 0
        guessClass = -1
        label1 = 0
        ks = []
        distancesArr = self.euclidian_dsitanceList(x, X)
        distancesArr.sort()
        for p in range(int(k)):
            minimum = min(distancesArr)
            ks.append((minimum[0], minimum[1]))
            distancesArr.remove(minimum)
        for q in range(len(ks)):
            if ks[q][1] == 0:
                label0 += 1
            else:
                label1 += 1
        if label0 >= label1:
            guessClass = 0
        else:
            guessClass = 1
        return guessClass


    def knn(self,XTest, X, k):
        #X is the training data
        guessList =[]
        guessedLabel = -1
        #print(XTest)
        for row in XTest:
            guessedLabel = self.guessLabel(row, X, k)
            guessList.append(guessedLabel)
        return guessList



    def predict(self, XTest, k=1, data="nX"):
        start_time = time.time()
        predictionArr =[]
        #print(XTest)
        if data == "nX":
            self.predictionArr = self.knn(XTest, self.nX, k)
        elif data == "pX":
            XTest = pca(XTest)
            self.predictionArr = self.knn(XTest, self.pX, k)
        else:
            XTest=fld(XTest, training=False)
            self.predictionArr = self.knn(XTest, self.fX, k)
        #print(self.predictionArr)
        self.totalTime= time.time() - start_time
        return  self.predictionArr

    def performanceCurve(self, X_test,ytest, K, data="fX"):
        k = np.arange(1, K+1, 1).tolist()
        accuracy_list = []
        for i in k:
            array_prediction=self.predict(X_test, i, data)
            accuracy_list.append(accuracy_score(ytest,array_prediction,showResults=False))
        #print(accuracy_list)
        print("Maximum Accuracy is obtained when K=", k[np.argmax(accuracy_list)])
        print("The accuracy associated with that K = ", np.amax(accuracy_list))
        plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
        plt.plot(k, accuracy_list)
        plt.legend(loc='best')
        plt.xlabel('K')
        plt.ylabel('Accuracy')
        plt.show()

