#python3

import os
import time
import numpy as np
#from sklearn import svm2
#import svm


#Library for plotting the output and saving it to file
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'

className = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

#print(dir(xTrain))

#print(xTest.nbytes)

#print(type(xTrain)) #ndarray [n dimensional array]

#print(dir(xTrain))
#print(xTrain)

plt.subplot(221)
plt.imshow(xTrain[10])
plt.axis('off')
plt.title(className[yTrain[10][np.array(0)]])
plt.subplot(222)
plt.imshow(xTrain[11])
plt.axis('off')
plt.title(className[yTrain[11][np.array(0)]])
plt.subplot(223)
plt.imshow(xTest[10])
plt.axis('off')
plt.title(className[yTest[10][np.array(0)]])
plt.subplot(224)
plt.imshow(xTest[11])
plt.axis('off')
plt.title(className[yTest[11][0]])
plt.savefig(baseDir + 'svm0.png')
plt.clf()


xVal = xTrain[49000:, :].astype(np.float)
yVal = np.squeeze(yTrain[49000:, :])
xTrain = xTrain[:49000, :].astype(np.float)
yTrain = np.squeeze(yTrain[:49000, :])
yTest = np.squeeze(yTest)
xTest = xTest.astype(np.float)



print("haris ", xTrain.nbytes)
print(xTrain.shape)

meanImage = np.mean(xTrain, axis = 0)
xTrain -= meanImage
xVal -= meanImage
xTest -= meanImage


xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
xVal = np.reshape(xVal, (xVal.shape[0], -1))
xTest = np.reshape(xTest, (xTest.shape[0], -1))

xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])

#print(xVal.size)
#print(xTrain)

#numClasses = np.max(yTrain) + 1

#classifier = svm.svm(xTrain.shape[1], numClasses)

#if classifier.W is not None:
#    tmpW = classifier.W[:-1, :]
#    tmpW = tmpW.reshape(32, 32, 3, 10)
