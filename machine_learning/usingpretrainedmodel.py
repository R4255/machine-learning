#when we have small amount of datasets 
#we saw that accuracy was 70% approx
#to get a high accuracy using small amount of datasets itself we use redimate layers already trained with thousands and thousands of images

import os 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import tensorflow_datasets as tfds 
tfds.disable_progress_bar()
#split the data into 80% training , 10% testing ,10% validation
(raw_train,raw_validation,raw_test),metadata=tfds.load(
    'cats_vs_dogs',
    split=['train[:%80]','train[80%:90%]','train[90%]'],
    with_info=True,
    as_supervised=True,
)
#now we create a function object that we can use to get the labels
get_label_name=metadata.features['label'].int2str
#display 2 images from dataset 
for image,label in raw_train.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
#resource module is not available in windows operating system 
IMG_SIZE=160
#function to resize all the images to 160*160
def format_example(image,label):
    image=tf.cast(image,tf.float32)
    image=(image/127.5)-1
    image=tf.image.resize(image,(IMG_SIZE,IMG_SIZE))
    return image,label
#we can apply this function to all our images using the .map() function 
train=raw_train.map(format_example)
validation=raw_validation.map(format_example)
test=raw_test.map(format_example)
for image,label in train.take(2):
    plt.figure()
    plt.imshow()
    plt.title(get_label_name(label))
#we will now shuffle and batch the images 
BATCH_SIZE=32
SHUFFLE_BUFFER_SIZE=1000
train_batches=train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches=validation.batch(BATCH_SIZE)
test_batches=test.batch(BATCH_SIZE)
#Now if we look at the shape of an original image vs the new image we will see it has been changed.
for img,label in raw_train.take(2):
    print('original shape:',img.shape)
for img,label in train.take(2):
    print('new shape',img.shape)
#mobinet v2 developed by google has been trained by 1.4 million photos , 1000 classes
#this is the model that we will use here
#we will use the base and starting layers not the ending layers
IMG_SHAPE=(IMG_SIZE,IMG_SIZE,3)
#creating the base model from mobinet v2
base_model=tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
base_model.summary()
feature_batch=base_model(image)
print(feature_batch.shape)
#the term freezing means disabling the training property of a layer
base_model.trainable=False
base_model.summary()
#adding the classifier
global_average_layer=tf.keras.layers.GlobalAveragePooling2D()
#since we have only 2 class to predict we will have a single dense neuron in the prediction layer
prediction_layer=keras.layers.Dense(1)
#COMBINING THE LAYERs
model=tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()
#training the model 
#we will use a very small learning rate to ensure that the model does not have any major changes made to it
base_learning_rate=0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
initial_epochs=3
validation_steps=20
loss0,accuracy0=model.evaluate(validation_batches,steps=validation_steps)
history=model.fit(train_batches,epochs=initial_epochs,validation_data=validation_batches)
acc=history.history['accuracy']
print(acc)  
model.save("dogs_vs_cats.h5")#we can save the model and reload it anytime in the future
new_model=tf.keras.models.load_model('dogs_vs_cats.h5')