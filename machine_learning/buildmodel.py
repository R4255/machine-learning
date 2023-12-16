#mainly we will work with two premade library here
#DNNClassifier
#linearclassifier
import tensorflow as tf
classifier=tf.estimator.DNNClassifier(
    feature_columns=my_feature_column,
    #we will make here two hidden layers of 30 and 10 nodes each
    hidden_units=[30,10],
    n_classes=3
)
#training the model
classifier.train(
    input_fn=lambda:input_fn(train,train_y,training=True),
    steps=5000
)