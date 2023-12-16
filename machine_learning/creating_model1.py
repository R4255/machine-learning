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
linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)#train
result=linear_est.evaluate(eval_input_fn)#get the model stats by testing on testing data
clear_output()#clears the console output
print(result['accuracy'])#the result variable is simply a dict of stats about our model 
