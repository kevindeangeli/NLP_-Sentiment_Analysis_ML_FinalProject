'''
Created by Kevin De Angeli & Hector D. Ortiz-Melendez
Date: 2019-11-24
'''

from GaussianClassifiers import *
from KNN import *
import matplotlib.pyplot as plt
from K_means import *

import time #For computing time
from sklearn.metrics import confusion_matrix #To compute the confusion matrix.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #Bag-of-words/TF-IDF (Feature Extraction)
from sklearn import svm #Support Vector Machine
from sklearn.model_selection import train_test_split #To split the data into testing/training
from sklearn.metrics import accuracy_score #To compute the accuracy of each model
from sklearn import tree #Decision Trees
from sklearn.neural_network import MLPClassifier #For BPNN
from sklearn.model_selection import KFold #to split the data
from sklearn.ensemble import RandomForestClassifier #RnadomForest
from sklearn.linear_model import LogisticRegression #For Classifier fussion
from sklearn.naive_bayes import GaussianNB # For classifier fussion
from sklearn.ensemble import VotingClassifier  #For classifier fussion.
from sklearn.utils import shuffle #To shuffle the data when we merge the three subsets.

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
    model = mpp(2)
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


def crossValidationExample(X,Y, classifierClass):
    #This is an example of how you do k-fold cross validation.
    #X, Y = readData(data)
    timeStart = time.time()
    X = BoW(X)
    kf = KFold(n_splits=10) # Split the data into 10 subsets.
    kf.get_n_splits(X)
    accuracy = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        clf = classifierClass
        #clf = tree.DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_predict))
    print("--- %s seconds ---" % (time.time() - timeStart))
    #print(accuracy)
    print(np.mean(accuracy))



def randomForest(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier( random_state=0)
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    print(accuracy_score(y_test, y_predict))

def mergeDatasets(data):
    x1, y1 = readData(data[0])
    x2, y2 = readData(data[1])
    x3, y3 = readData(data[2])
    X_all = np.concatenate([x1,x2,x3])
    Y_all = np.concatenate([y1,y2,y3])
    X_all, Y_all = shuffle(X_all, Y_all, random_state=0) #Shuffle data
    return X_all, Y_all

def plotConfusionMatrix(y_predict, y_test):
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    labels = [1, 0]
    cm = confusion_matrix(y_test, y_predict, labels)
    print(cm)
    print(type(cm))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

def plotAmazonCM():
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    labels = [1, 0]
#    cm = confusion_matrix(y_test, y_predict, labels)
    
    # Amazon
    AmazonCMbow = [83, 78, 20, 149]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(AmazonCMbow)
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    AmazonCMtf = [108, 53, 30, 139]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(AmazonCMtf)
    #plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def NeuronsVSLayersVsAccuracy3D(X_train, X_test, y_train, y_test):
    #Note: The array of neurons and the array of hl should be the same
    #The program can be modified so it can take arbitrary numbers.
    neurons = np.arange(1, 10)  # Controls number of neurons in all layers.
    hl = np.arange(1,10)  # Controls number of layers
    network = []
    #network.append(784)
    #test = list(test_data)
    #network.append(1)
    accuracy =  np.zeros([hl.shape[0],neurons.shape[0]])
    for i in hl:
        network.append(1)
        for n in neurons:
            for k in range(len(network)):
                network[k] = n
            print("Network Architecture Being used: ",network)
            clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=tuple(network), random_state=1)
            clf.fit(X_train, y_train)
            y_guess = clf.predict(X_test)
            [coef.shape for coef in clf.coefs_]
            acc= accuracy_score(y_test, y_guess)
            acc = np.round_(acc,decimals=2)
            accuracy[hl.shape[0]-i][n-1] = acc


    print(accuracy)

    #Note: The heat map fuction displays the graph in the exact order as the
    #Matrix. So I had to populate the matrix in the opposite order
    left = neurons[0] - .5  # Should be set so that it starts a the first point in the array -.5
    right = neurons[-1] + .5  # last number of the array +.5
    bottom = hl[0] - .5
    top = hl[-1] + .5
    extent = [left, right, bottom, top]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    im = ax.imshow(accuracy, extent=extent, interpolation='nearest')
    ax.set(xlabel='Number of Neurons', ylabel='Number of Layers')

    #Label each square in the heat map:
    for i in range(len(hl)):
        for j in range(len(neurons)):
            text = ax.text(j + 1, i + 1, accuracy[accuracy.shape[0]-i-1,j],
                           ha="center", va="center", color="w")

    plt.show()


def classifierFussion(data):
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    for dat in data:
        X, Y = readData(dat)
        Xv1 = BoW(X)
        Xv2 = TF_IDF(X)
        featEx = [Xv1, Xv2, 'BoW', 'TF_IDF']
        count = 1
        for Xv in featEx[0:2]:
            count += 1
            X_train, X_test, y_train, y_test = splitData(Xv, Y)
            eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
            timeStart = time.time()
            eclf1 = eclf1.fit(X_train, y_train)
            y_predict = eclf1.predict(X_test)

            print('Data:', dat, '\nFeature Extraction:', featEx[count],accuracy_score(y_test, y_predict))
            print("--- %s seconds ---" % (time.time() - timeStart))
            print(" ")

    X, Y = mergeDatasets(data)
    Xv1 = BoW(X)
    X_train, X_test, y_train, y_test = splitData(Xv1, Y)
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    timeStart = time.time()
    eclf1 = eclf1.fit(X_train, y_train)
    y_predict = eclf1.predict(X_test)
    print('Data: All-Three', '\nFeature Extraction:BoW', accuracy_score(y_test, y_predict))
    print("--- %s seconds ---" % (time.time() - timeStart))
    print(" ")

    Xv2 = TF_IDF(X)
    X_train, X_test, y_train, y_test = splitData(Xv2, Y)
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    timeStart = time.time()
    eclf1 = eclf1.fit(X_train, y_train)
    y_predict = eclf1.predict(X_test)
    print('Data: All-Three', '\nFeature Extraction:= TF-IDF', accuracy_score(y_test, y_predict))
    print("--- %s seconds ---" % (time.time() - timeStart))
    print(" ")



def gaussianPriorProbsAnalysis(data, bow = True):
    if len(data) == 3:
        X, Y = mergeDatasets(data)  # all three combined
    else:
        X, Y = readData(data)  # one dataset at the time

    if bow:
        X = BoW(X)
    else:
        X= TF_IDF(X)

    X_train, X_test, y_train, y_test = splitData(X, Y)
    priors=np.linspace(0.1, 1, 100) #log(0) DNE
    ws = [0,0]
    acc = []
    for w in priors:
        ws[0] = w
        ws[1] = 1-w+.000001
        gaussian = mpp(1)
        gaussian.pw = ws
        gaussian.fit(X_train, y_train)
        prediction = gaussian.predict(X_test)
        acc.append(accuracy_score(y_test, prediction))

    print(acc)
    plt.figure(num=None, figsize=(8, 8), dpi=100, facecolor='w', edgecolor='k')
    plt.plot(priors, acc)
    plt.xlabel(xlabel='W1 (Prior Probability)')
    plt.ylabel(ylabel='Accuracy')
    plt.savefig('Images/priorProbs/example.png')

    #plt.show()

def knnStudy(dat, X, Y, ks, ke, merged=False):
        Xv1 = BoW(X)
        Xv2 = TF_IDF(X)
        featEx = [Xv1, Xv2, 'BoW', 'TF_IDF']
        count = 1
        
        print('TPR', 'FPR')

        for Xv in featEx[0:2]:
            count += 1
            X_train, X_test, y_train, y_test = splitData(Xv, Y)
#            accSVM = SVM(X_train, X_test, y_train, y_test)
#            accBPNN = BPNN(X_train, X_test, y_train, y_test)
#            accDT = DecisionTree(X_train, X_test, y_train, y_test)
            
            if merged == False:
                print('Data:', dat, '\nFeature Extraction:', featEx[count],"\n")
            else:
                print('Data: Merged Data ', '\nFeature Extraction:', featEx[count], '\nKNN', "\n")
            

            for k in range(ks,ke+1):

                start = time.time()
                
                Acc, TPR, TNR = knn(X_train, X_test, y_train, y_test, k)

                end = time.time()


def showConfusionMatrix(data, classifierClass):
    X, Y = readData(data)  # one dataset at the time
    labels = [1, 0]

    #X = TF_IDF(X)

    print("BOW")
    X = BoW(X)
    X_train, X_test, y_train, y_test = splitData(X, Y)

    classifier = classifierClass
    classifier.fit(X_train,y_train)
    y_predict=classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predict, labels)
    #Output is in this format: tn, fp, fn, tp
    print(cm)
    print(" ")
    print("TFIDF")
    X, Y = readData(data)  # one dataset at the time
    X = TF_IDF(X)
    X_train, X_test, y_train, y_test = splitData(X, Y)
    classifier = classifierClass
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_predict)
    # Output is in this format: tn, fp, fn, tp
    print(cm)



###########################################################################
####################### Main Starts Here  #################################
###########################################################################

def main():
    amazon= 'Data/amazon_cells_labelled.txt'
    imdb= 'Data/imdb_labelled.txt'
    yelp = 'Data/yelp_labelled.txt'
    data = [amazon, imdb, yelp]


    #threeVsAll(data) # Function from Milestone 3 to compute initial results
    #gaussian(imdb) #Case 1 works and gives bad accuracy (50%). Case II, and III don't work because of singular matrix when taking the inverse.
    #crossValidationExample(amazon, classifierClass=tree.DecisionTreeClassifier()) #Give the dataset and the classifier ;)
    #gaussianPriorProbsAnalysis(data, False) #False for TFIDF, True for BoW
    #classifierFussion(data)


    '''
    #This is to make the accuracy tables. Make sure to check what type of 
    #features are being used in the function crossValidationExample (bow or tf-dif).

    #X,Y = readData(yelp) # one dataset at the time
    X, Y = mergeDatasets(data) #all three combined 
    crossValidationExample(X=X,Y=Y, classifierClass=RandomForestClassifier( random_state=0)) #Give the dataset and the classifier ;)
    


    
    #This was used to plot the confusion matrix
    
    X,Y = readData(amazon) # one dataset at the time
    #X, Y = mergeDatasets(data) #all three combined
    X = BoW(X)
    #X = TF_IDF(X)
    X_train, X_test, y_train, y_test = splitData(X, Y)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5), random_state=1)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    plotConfusionMatrix(prediction, y_test)
    



    '''
    #This code is to create the heat map with the neuronsVsLayers.
    
#    X,Y = readData(amazon) # one dataset at the time
#    X = TF_IDF(X)
#    X_train, X_test, y_train, y_test = splitData(X, Y)
#    NeuronsVSLayersVsAccuracy3D(X_train, X_test, y_train, y_test)

    kstart  = 1
    kend    = 10
    # KNN individual data
    for dat in data:
        X, Y = readData(dat)
        knnStudy(dat, X, Y, kstart, kend)
    
    # KNN merged data
    X, Y = mergeDatasets(data)
    knnStudy('ignore', X, Y, kstart, kend, True)

    '''
    #This code runs KNN

    X,Y = readData(yelp) # one dataset at the time
    #X, Y = mergeDatasets(data) #all three combined
    X = TF_IDF(X)
    X_train, X_test, y_train, y_test = splitData(X, Y)
    k_means = Kmeans()
    k_means.fit(X_train , y_train, iterationsLimit= 1000)
    y_predict = k_means.predict(y_test)
    print(accuracy_score(y_test, y_predict))
    '''

    #showConfusionMatrix(amazon, svm.SVC(gamma='scale'))
    #showConfusionMatrix(amazon, MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,5), random_state=1))
    #showConfusionMatrix(amazon, RandomForestClassifier( random_state=0))
    '''
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GaussianNB()
    showConfusionMatrix(amazon,  VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard'))
    '''
    showConfusionMatrix(amazon,  mpp())




if __name__ == "__main__":
    main()
