from sklearn import svm
import numpy as np
import h5py
import pickle
import argparse

import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

f = h5py.File('images.h5', 'r') 
X = np.array(f['faceFeatures'])
y = np.array(f['emotion'])

clf = svm.SVC()
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
print(X.shape)
p = np.random.permutation(len(X))
X = X[p]
y = y[p]



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



ap = argparse.ArgumentParser()
ap.add_argument('-v', '--visualize', action='store_true')
args = vars(ap.parse_args())

if (not args["visualize"]):
  clf.fit(X, y)

  print(clf.score(X, y))

  with open('emotion_classifier.pkl', 'wb') as fid:
    pickle.dump(clf, fid)


else:
  train_split = 0.8

  num_images = y.shape[0]
  X_train = X[0:int(round(train_split*num_images))]
  y_train = y[0:int(round(train_split*num_images))]
  X_test = X[int(round(train_split*num_images))+1:-1]
  y_test = y[int(round(train_split*num_images))+1:-1]


  clf.fit(X_train,y_train)

  print(clf.score(X_test,y_test))

  y_pred = clf.predict(X_test)

  cnf_matrix = confusion_matrix(y_test, y_pred)
  np.set_printoptions(precision=2)

  class_names = ["neutral", "anger",
  	"joy",
  	"sadness",
  	"fear",
  	"disgust",
  	"shame"]

  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix(cnf_matrix, classes=class_names,
                        title='Confusion matrix, without normalization',
                        normalize=True)

  plt.show()