'''
Created by Kevin De Angeli & Hector D. Ortiz-Melendez
Date: 2019-11-24
'''

from GaussianClassifiers import *
from KNN import *
from Distances import *

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #Bag-of-words/TF-IDF (Feature Extraction)
from sklearn import svm #Support Vector Machine
from sklearn.metrics import accuracy_score #To compute the accuracy of each model
from sklearn.model_selection import train_test_split #To split the data into testing/training
from sklearn import tree #Decision Trees
from sklearn.neural_network import MLPClassifier #For BPNN
from sklearn.model_selection import KFold #to split the data

def readData(path):
    file = open(path, "r")
    Y = []
    X = []
    while True:
        line = file.readline()
        if not line:
            break
        X.append(line[0:-3])
        Y.append(line[-2])
    Y = [int(i) for i in Y]  # make labels int instead of str

    return np.array(X), np.array(Y)

def getDataInfo(X, labels = False):
    print("************************")
    print("Example first 3 rows: \n")
    print(X[0:3])
    print("\nNumber of elements: ", len(X))
    if labels:
        print("Number of entries with label 1: ", np.sum(X==1))
        print("Number of entries with label 0: ", len(X)-np.sum(X==1))
    print("************************\n")



def BoW(X):

    vectorizer = CountVectorizer()
    Xv = vectorizer.fit_transform(X)
    X_bow = Xv.toarray()

#    numpy.set_printoptions(threshold=sys.maxsize)
#    with open('junk', 'w') as f:
#        for item in my_list:
#            f.write("%s\n" % item)
    return X_bow

def TF_IDF(X):
    vectorizer = TfidfVectorizer()
    Xv = vectorizer.fit_transform(X)
    X_TFIDF = Xv.toarray()

    return X_TFIDF
    
def splitData(X,y,testSize=0.33):
    #This is the most basic way to split data:  77% Trainning and 33% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    return X_train, X_test, y_train, y_test
    
def SVM(X_train, X_test, y_train, y_test):
    clf = svm.SVC(gamma='scale')
    clf.fit(X_train, y_train)
    y_guess = clf.predict(X_test)
    # get support vectors
    clf.support_vectors_
    # get indices of support vectors
    clf.support_
    # get number of support vectors for each class
    clf.n_support_
    return accuracy_score(y_test, y_guess)

def BPNN(X_train, X_test, y_train, y_test, hiddenLayer = (5,2)):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hiddenLayer, random_state=1)
    clf.fit(X_train, y_train)
    y_guess = clf.predict(X_test)
    [coef.shape for coef in clf.coefs_]
    return accuracy_score(y_test, y_guess)

def DecisionTree(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_guess = clf.predict(X_test)
    return accuracy_score(y_test, y_guess)

def gaussian(data):
    X, Y = readData(data)
    Xv1 = BoW(X)
    X_train, X_test, y_train, y_test = splitData(Xv1, Y)
    model = mpp(1)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    print("Gaussian Accuracy:",accuracy_score(y_test, prediction))

def threeVsAll(data):
    #This is the function of Milestone 2 used to compute some initial results.
    for dat in data:
        X, Y = readData(dat)
        Xv1 = BoW(X)
        Xv2 = TF_IDF(X)
        featEx = [Xv1, Xv2, 'BoW', 'TF_IDF']
        count = 1
        for Xv in featEx[0:2]:
            count += 1
            X_train, X_test, y_train, y_test = splitData(Xv, Y)
            accSVM = SVM(X_train, X_test, y_train, y_test)
            accBPNN = BPNN(X_train, X_test, y_train, y_test)
            accDT = DecisionTree(X_train, X_test, y_train, y_test)
            print('Data:', dat, '\nFeature Extraction:', featEx[count], '\nSVM', accSVM, 'BPNN', accBPNN, 'DT', accDT,
                  "\n")


#    pdb.set_trace()

def crossValidationExample(data):
    #This is an example of how you do k-fold cross validation.
    X, Y = readData(data)
    X = TF_IDF(X)
    kf = KFold(n_splits=10) # Split the data into 10 subsets.
    kf.get_n_splits(X)
    accuracy = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_predict))

    print(accuracy)
    print(np.mean(accuracy))



def main():
    imdb= 'Data/imdb_labelled.txt'
    amazon= 'Data/amazon_cells_labelled.txt'
    yelp = 'Data/yelp_labelled.txt'
    data = [imdb, amazon, yelp]

    #threeVsAll(data) # Function from Milestone 1 to compute initial results
    gaussian(imdb) #Case 1 works and gives bad accuracy (50%). Case II, and III don't work currently.
    #crossValidationExample(amazon)



if __name__ == "__main__":
    main()
