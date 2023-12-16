from __future__ import absolute_import,division,print_function,unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
#from six.moves import urllib
#import tensorflow.compat.v2.feature_column as fc #feature column 
#required when wanted to create linear regression or tensor flow model 
import tensorflow as tf
dftrain=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')#training data
dfeval=pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')#testing data
#print(dftrain.head())
y_train=dftrain.pop('survived')
y_eval=dfeval.pop('survived')
# print(dftrain.head())
# #the pd.read_csv method returns us a new pandas dataframe.we can imagine dataframe like a table
# print(y_train)
# print(dftrain.loc[0],y_train.loc[0])
# print("---------------")
# print(dftrain["age"])
# print("-------------")
# print(dftrain.describe())
# print(dftrain.shape)
# #gives us the no. of entries and the features
# plt.hist(dftrain['age'],bins=20)
# plt.show()
categorical_column=['sex','n_siblings_spouses','parch','class','deck','embark_town','alone']
numerical_column=['age','fare']
feature_column=[]
for feature_name in categorical_column:
    vocabulary=dftrain[feature_name].unique()#gets a list of all unique values from given feature column
    feature_column.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))
    #feature_column it is used to create numerical feature column
#feature column , they are just what we want to feed to our linear model to make predictions
for feature_name in numerical_column:
    feature_column.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))
#print(feature_column)
dftrain["sex"].unique()
#we train our model , we give the data to the model in batches 
#here we give the model 32 info at a time
#epoch is simply how many times the same data has passed through the model , we pass the pass the data in different orders and we predict the pattern
#we create a input function 
#input function defines how our dataset will be converted into batches at each epoch

import tensorflow as tf
def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_function():#linear function , this will be returned
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)#split the dataset into batches of 32 and repeat process for the given number of epochs
        return ds
    return input_function
train_input_fn=make_input_fn(dftrain,y_train)
eval_input_fn=make_input_fn(dfeval,y_eval,num_epochs=1,shuffle=False)
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_column)
linear_est.train(train_input_fn)#train
result=linear_est.evaluate(eval_input_fn)#get the model stats by testing on testing data
clear_output()#clears the console output
# print(result['accuracy'])#the result variable is simply a dict of stats about our model 
# print(result)
print("-----------------------")
result=list(linear_est.predict(eval_input_fn))
#print(result)
print(dfeval.loc[4])
print(y_eval.loc[4])
print(result[4]['probabilities'][1])
