#the three main properties of each convolution layer (input size,the number of filters,sample size of filters)
#convolution neural network will scan through our entire image , it will pick up features and find features in the image
#and then based on the features that exist in the image will pass that actually to a dense neural network or dense classifier
#dense neural network will search for the pattern at a specific portion of the image
#filter is just some pattern of pixels 
#filters are the one that is going to be trained 
#filter is what will be found by the neural network 
#padding is adding an extra layer of row and column to give the same output matrix as that of the input matrix 
#stride , is how much we have moved the sample box everytime 
#pooling operation is just taking specific values from a sample of the output feature map 
import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras import models
from keras import datasets,layers,models
import matplotlib.pyplot as plt
#load and split the data
(train_images,train_labels),(test_images,test_labels)=datasets.cifar10.load_data()
#normalize the pixel value to be between 0 and 1
train_images,test_images=train_images/255.0,test_images/255.0
class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
IMG_INDEX=1#change this to look at other images
plt.imshow(train_images[IMG_INDEX],cmap=plt.cm.binary)
#plt.cm: This is the submodule in Matplotlib that provides access to various colormaps.
#binary: This is the specific colormap chosen. As mentioned earlier, it represents a range of values from dark (usually black) to light (usually white).
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.imshow()
#architecture of convolution neural network 
#we stack all the min , max , avg pooling layers together
model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))#this is just done to reduce the dimensionality
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))#64 neurons + the activation specified
model.add(layers.Dense(10))#the output layer has 10 neurons
#now training the model 
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=model.fit(train_images,train_labels,epochs=4,validation_data=(test_images,test_labels))
