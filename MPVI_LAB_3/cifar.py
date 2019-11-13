import os
import time
import numpy as np


#Library for plotting the output and saving it to file
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#Load the CIFAR10 dataset
from keras.datasets import cifar10
baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'

className = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()

#print(xTrain)

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

print(xVal.size)
