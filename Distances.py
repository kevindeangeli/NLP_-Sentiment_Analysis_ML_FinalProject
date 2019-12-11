'''
Created by Kevin De Angeli and Hector D. Ortiz-Melendez
Date: 2019-12-03
'''

import numpy as np
import pdb
import math

def euc2(x, y):
    # calculate squared Euclidean distance
    # check dimension
    assert x.shape == y.shape
    diff = x - y
    return np.dot(diff, diff)

def mah2(x, y, Sigma):
    # calculate squared Mahalanobis distance
    # check dimension
    assert x.shape == y.shape and max(x.shape) == max(Sigma.shape)
    diff = x - y
    return np.dot(np.dot(diff, np.linalg.inv(Sigma)), diff)
    
def euclideanDistance(xTest,xTrain,length):

    distance = 0

    for x in range(length):
        distance += pow((xTest[x] - xTrain[x]),2)

#        pdb.set_trace()

    return math.sqrt(distance)
