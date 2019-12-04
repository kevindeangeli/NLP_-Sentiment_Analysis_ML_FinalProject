from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer #Bag-of-words/TF-IDF (Feature Extraction)
from sklearn.model_selection import train_test_split #To split the data into testing/training
from sklearn.ensemble import RandomForestClassifier #RnadomForest
import matplotlib.pyplot as plt



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


def TF_IDF(X):
    vectorizer = TfidfVectorizer()
    Xv = vectorizer.fit_transform(X)
    X_TFIDF = Xv.toarray()

    return X_TFIDF


def splitData(X,y,testSize=0.33):
    #This is the most basic way to split data:  77% Trainning and 33% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
    return X_train, X_test, y_train, y_test



imdb = 'Data/imdb_labelled.txt'
amazon = 'Data/amazon_cells_labelled.txt'
yelp = 'Data/yelp_labelled.txt'
data = [imdb, amazon, yelp]


X, Y = readData(amazon)
X = TF_IDF(X)
X_train, X_test, y_train, y_test = splitData(X, Y)

clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)



labels = [1,0]

cm = confusion_matrix(y_test, y_predict, labels)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

