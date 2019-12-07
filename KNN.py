'''
Created by Kevin De Angeli
Date: 2019-12-03
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statistics import mean
#from scipy import integrate
from numpy import array
from numpy import cov
#from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import sympy as sym
from sympy.matrices import MatrixSymbol, Transpose
from sympy.functions import transpose
from sympy import symbols, Eq, solve, nsolve
import math
from mpmath import *
import pdb
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from Distances import *

import operator
import time

def knn(nXtrain, nXtest, y_train, y_test, k):

    
    A0 = 1
    A1 = 1

    knn         = []

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    rowTe       = len(nXtest)
    columns     = len(nXtest[0])-1

    #    classes     = nXtest[:,-1]
    #    guesses     = nXtrain[:,-1]
    classes     = y_test
    guesses     = y_train


    for row in range(rowTe):
        

        distances       = []

        for x in range(len(nXtrain)):

            diff = nXtest[row] - nXtrain[x]
            
            dist = np.dot(diff, diff)

            distances.append((dist,guesses[x]))
        

        #print(distances)
        distances.sort(key=operator.itemgetter(0))
        
        for K in range(k):
            knn.append(distances[K][1])


        label0 = 0
        label1 = 0
       
        for K in knn:
           if K == 0:
               label0 += 1
           else:
               label1 += 1

        if (A0 == 1 and A1 == 1):
           if label0 > label1:
               guess = 0
           else:
               guess = 1
        else:
           # prior chosen
           ca0      = label0 * A0 / k
           ca1      = label1 * A1 / k

           if ca0 > ca1:
               guess = 0
           else:
               guess = 1
         

        if   (guess == 1 and classes[row] == 1):
           # True Positive
           TP += 1
            
        elif (guess == 0 and classes[row] == 1):
           # False Negative
           FN += 1
            
        elif (guess == 1 and classes[row] == 0):
           # False Positive
           FP += 1

        elif (guess == 0 and classes[row] == 0):
           # True Negative
           TN += 1


        knn     = []


    Acc = (TP + TN) / (TP + TN + FP + FN)
    TPR = TP / (TP + FN)    # sensitivty
    FPR = FP / (FP + TN)

    TNR = 1 - FPR           # specificity
    FNR = 1 - TPR

    #    print('CM: TN, FP, FN, TP = ', TN, FP, FN, TP)

    #    print('kNN Accuracy: ' , Acc)
    #    print('TPR,TNR',TPR,TNR)
    print('CM: TN, FP, FN, TP = ', TN, FP, FN, TP)
    print('K:',k,'Accuracy:' , Acc)
    print('sensitivity,specificity',TPR,TNR)

#    if (A0 == 1 and A1 == 1):
#       if k == 1:
#           print('CM: TN, FP, FN, TP = ', TN, FP, FN, TP)
#           print('K:',k,'Accuracy:' , Acc)
#           print('sensitivity,specificity',TPR,TNR)
#       elif k == 10:
#           print('CM: TN, FP, FN, TP = ', TN, FP, FN, TP)
#           print('K:',k,'Max Accuracy:' , Acc)
#           print('sensitivity,specificity',TPR,TNR)


          
    return Acc, TPR, TNR
