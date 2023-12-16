#neural network task is to provide classification and prediction for us 
#optimizer is just the algo that does the gradient descent and back propagation for us 
import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt
fashion_mnist=keras.datasets.fashion_mnist#loading the dataset here
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()#split into testing and training
#train_images.shape()
print(type(train_images))
#we have 60000 images that are made up of 28*28 pixels(784)
print(train_images[0,23,23])#this is one pixel (row 23 column 23)
print(train_labels[:10])
class_names=['T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
plt.figure()
plt.imshow(train_images[100])
plt.colorbar()
plt.grid(False)
plt.show()
#data preprocessing 
train_images=train_images/255.0
test_images=test_images/255.0
#we do this just to make it easier for the model to process
model=keras.Sequential([#sequential just means passing the data from left to right or passing the data sequentially 
    keras.layers.Flatten(input_shape=(28,28)),#input layer, j
    keras.layers.Dense(128,activation='relu'),#hidden layer,relu means rectified linear unit
    #dense simply means that neurons in a layer are connected to all the nodes in the previous layer
    keras.layers.Dense(10,activation='softmax')#output layer
    #output layers have that many neuron as we want the classes 
])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10)
#we are fitting it into the training data
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=1)
print('test accuracy:',test_acc)
#predictions=model.predict(test_images)
#if we want to pass just one item in predict we have do model.predict([test_images[0]]) because it expects only array   
#print(predictions)
predictions=model.predict(test_images)
print(class_names[np.argmax(predictions[0])])
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()