'''
Created by Kevin De Angeli & Hector D. Ortiz-Melendez
Date: 2019-11-24
'''

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


class mpp:
    def __init__(self, case=1):
        # init prior probability, equal distribution
        # self.classn = len(self.classes)
        # self.pw = np.full(self.classn, 1/self.classn)

        # self.covs, self.means, self.covavg, self.varavg = \
        #     self.train(self.train_data, self.classes)
        self.case_ = case
        self.pw_ = None

    def fit(self, Tr, y, fX=False):
        # derive the model
        self.covs_, self.means_ = {}, {}
        self.covsum_ = None

        self.classes_ = np.unique(y)  # get unique labels as dictionary items
        self.classn_ = len(self.classes_)

        for c in self.classes_:
            arr = Tr[y == c]
            self.covs_[c] = np.cov(np.transpose(arr))
            self.means_[c] = np.mean(arr, axis=0)  # mean along rows
            if self.covsum_ is None:
                self.covsum_ = self.covs_[c]
            else:
                self.covsum_ += self.covs_[c]

        if fX==False:
            # used by case II
            self.covavg_ = self.covsum_ / self.classn_

            # used by case I
            self.varavg_ = np.sum(np.diagonal(self.covavg_)) / len(self.classes_)
        else:
            self.covavg_ = np.std(Tr)
            self.varavg  = np.var(Tr)

    def predict(self, T):
        # eval all data
        y = []
        disc = np.zeros(self.classn_)
        nr, _ = T.shape

        if self.pw_ is None:
            self.pw_ = np.full(self.classn_, 1 / self.classn_)

        for i in range(nr):
            for c in self.classes_:
                if self.case_ == 1:
                    edist2 = euc2(self.means_[c], T[i])
                    disc[c] = -edist2 / (2 * self.varavg_) + np.log(self.pw_[c])
                elif self.case_ == 2:
                    mdist2 = mah2(self.means_[c], T[i], self.covavg_)
                    disc[c] = -mdist2 / 2 + np.log(self.pw_[c])
                elif self.case_ == 3:
                    mdist2 = mah2(self.means_[c], T[i], self.covs_[c])
                    disc[c] = -mdist2 / 2 - np.log(np.linalg.det(self.covs_[c])) / 2 \
                              + np.log(self.pw_[c])
                else:
                    print("Can only handle case numbers 1, 2, 3.")
                    sys.exit(1)
            y.append(disc.argmax())

        return y

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
    #gaussian(imdb) Case 1 works and gives bad accuracy (50%). Case II, and III don't work currently.
    #crossValidationExample(amazon)




if __name__ == "__main__":
    main()
