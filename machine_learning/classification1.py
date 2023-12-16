#classification is the process of separating the data points into different classes
from __future__ import absolute_import,division,print_function,unicode_literals
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#we are given here the info on sepal length,sepal width , petal length,petal width and then we are going to tell to what species 
#it belong in setosa,versicolor,virginica
csv_column_names=['sepallength','sepalwidth','petallength','petalwidth','species']
species=['setosa','versicolor','virginica']
train_path=tf.keras.utils.get_file("iris_training.csv",'https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv')
test_path=tf.keras.utils.get_file('iris_test.csv','https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv')
train=pd.read_csv(train_path,names=csv_column_names,header=0)#this means row 0 is the header (header=0 specifies this)
test=pd.read_csv(test_path,names=csv_column_names,header=0)
#we use keras , module inside of the tensorflow to grab on to the datasets and read them into a pandas dataframe
#train.head()
train_y=train.pop('species')
test_y=test.pop('species')
train.head()
print(train.shape)
def input_fn(features,labels,training=True,batch_size=256):
    #converting the inputs to a dataset
    dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    #we will shuffle and repeat if we are in the training mode
    if training:
        dataset=dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)
my_feature_columns=[]
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
classifier=tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    #we will make here two hidden layers of 30 and 10 nodes each
    hidden_units=[30,10],
    n_classes=3
)
#training the model
classifier.train(
    input_fn=lambda:input_fn(train,train_y,training=True),
    steps=5000#similar to epochs

)
eval_result=classifier.evaluate(input_fn=lambda:input_fn(test,test_y,training=False))
print('\nTest set accuracy:{accuracy:0.3f}\n'.format(**eval_result))
import tensorflow as tf
def input_fn(features,batch_size=256):
    #convert the inputs to a dataset without labels
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
#from_tensor_slices method takes a dictionary of features and creates a dataset with elements that are dictionaries containing the individual feature values
features=['sepallength','sepalwidth','petallength','petalwidth']
predict={}
print("please type numeric values as prompted")
for feature in features:
    valid=True
    while valid:
        val=input(feature+": ")
        if not val.isdigit():valid=False
    predict[feature]=[float(val)]
predictions=classifier.predict(input_fn=lambda:input_fn(predict))
for pred_dict in predictions:
    class_id=pred_dict['class_ids'][0]
    probability=pred_dict['probabilities'][class_id]
    print('prediction is "{}"({:.1f}%)'.format(species[class_id],100*probability))