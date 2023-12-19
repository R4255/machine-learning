# we will give the input and it will predict the next character in the sequence
import tensorflow as tf 
from keras.preprocessing import sequence
import keras
import os 
import numpy as np 
#dataset , we will use the romeo and juliet play here as the input 
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
#we can load our own dataset instead by
'''
from google.colab import files 
path_to_file=list(files.upload().keys())[0]
'''
text=open(path_to_file,'rb').read().decode(encoding='utf-8')
print('length of the text file : {} characters'.format(len(text)))
#the first 250 characters in the text are 
print(text[:250])

#encoding , we are going to encode each unique character as a different integer
#the set in python contains only unique elements
vocab=sorted(set(text))
#create a mapping from unique char to indices
char2idx={u:i for i,u in enumerate(vocab)}
idx2char=np.array(vocab)
def text_to_int(text):
    return np.array([char2idx[c] for c in text])
text_as_int=text_to_int(text)
print("text",text[:13])
print("encoded:",text_to_int(text[:13]))

# function that convert our numerical values to text

def int_to_text(ints):
    try:
        ints=ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])
print(int_to_text(text_as_int[:13]))

#CREATING THE TRAINING MODEL

'''Remember our task is to feed the model a sequence and have it return to us the next character. This means we need to split our text data from above into many shorter sequences that we can pass to the model as training examples. 

The training examples we will prepapre will use a *seq_length* sequence as input and a *seq_length* sequence as the output where that sequence is the original sequence shifted one letter to the right. For example:

```input: Hell | output: ello```

Our first step will be to create a stream of characters from our text data.'''
#// is the floor division 

seq_length=100
example_per_epoch=len(text)//(seq_length+1)
#creating the training examples/targets
char_dataset=tf.data.Dataset.from_tensor_slices(text_as_int)
#what it does mainly is convert the text to characters 

#we will now create batches of appropriate size
sequences=char_dataset.batch(seq_length+1,drop_remainder=True)
#we will drop the  last/remainder if it oversizes the given /required size

# now we will use these sequences of length 101 and split them input and output
def split_input_target(chunk):#example hello
    input_text=chunk[:-1]#hell
    target_text=chunk[1:]#ello
    return input_text,target_text
dataset=sequences.map(split_input_target)

for x, y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_text(x))
    print("\nOUTPUT")
    print(int_to_text(y))

#making the training batches
BATCH_SIZE=64
VOCAB_SIZE=len(vocab)
EMBEDDING_DIM=256#how big we want every single vector to represent our word in the embedding layer 
RNN_UNITS=1024
#it maintains a buffer in which it shuffles elements
BUFFER_SIZE=10000
data=dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)

#building the model 

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()
#we are feeding 64 entries that are each of 100 length into the model as its training data
#CREATING THE LOSS FUNCTION NOW

for input_example_batch,target_example_batch in data.take(1):
    example_batch_predictions=model(input_example_batch)
    print(example_batch_predictions.shape,"#(batch_size, sequence_length ,vocab_size)")
print(len(example_batch_predictions))
print(example_batch_predictions)
pred=example_batch_predictions[0]
print(len(pred))
print(pred)
# print("--------------")
# print(pred[0])
# print(len(pred[0]))
sampled_indices=tf.random.categorical(pred,num_shapes=1)
sampled_indices=np.reshape(sampled_indices,(1,-1))[0]
predicted_chars=int_to_text(sampled_indices)
print(predicted_chars) #this is what the model predicted for training sequence 1

def loss(labels,logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True)

# at this point we can think of our problem as a classification problem where the model predicts the probability of each unique
#letter coming next
#COMPILE THE MODEL 
model.compile(optimizer='adam',loss=loss)
#creating checkpoints ,we will setup and configure the model to save checkpoints as it trains.this will allow the model to load from a 
#certain checkpoint and continue training it 
checkpoint_dir='./training_checkpoints'#directory where checkpoints will be saved 
checkpoint_prefix=os.path.join(checkpoint_dir,"ckpt_{epoch}")#this is the name of the checkpoint files 
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)
#training the model 
history=model.fit(data,epochs=40,callbacks=[checkpoint_callback])
#loading the model 
#we will rebuild the model from a checkpoint using a batch_size of 1 so that we can feed one piece of text to the model and have it make 
#make a prediction 
model=build_model(VOCAB_SIZE,EMBEDDING_DIM,RNN_UNITS,batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1,None]))#one is saying expect the input one , none specifies we dont know the what next dimension will be
#we can load any checkpoint we want by specifing the exact file to load
checkpoint_num=10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_"+str(checkpoint_num)))
model.build(tf.TensorShape([1,None]))

#FINALLY GENERATING THE TEXT
def generate_text(model,start_string):
    num_generate=800
    #converting the start string to numbers(vectorization)
    input_eval=[char2idx[s] for s in start_string]
    input_eval=tf.expand_dims(input_eval,0)
    #empty string to store our results
    text_generated=[]
    #low temperature results in more predictable text
    #higher temperature results in more surprising text
    #experiment to find the best setting 
    temperature=1.0
    #batch size=1
    model.reset_states()
    for i in range(num_generate):
        predictions=model(input_eval)
        #remove the batch dimension 
        predictions=tf.squeeze(predictions,0)
        #using the categorical distribution to predict the character returned by the model 
        predictions=predictions/temperature
        predicted_id=tf.random.categorical(predictions,num_samples=1)[-1,0].numpy()
        #we pass the predicted ch as the next input to the model 
        #along with the previous hidden state
        input_eval=tf.expand_dims([predicted_id],0)
        text_generated.append(idx2char[predicted_id])
    return (start_string+''.join(text_generated))
inp=input("type the starting string:")
print(generate_text(model,inp))

