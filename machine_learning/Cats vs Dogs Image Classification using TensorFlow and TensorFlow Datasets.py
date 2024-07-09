import os 
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
keras=tf.keras
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
    