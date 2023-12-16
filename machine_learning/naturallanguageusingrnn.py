#word embedding are a vectorized representation of words in a given document that places words with similar meanings near each other.
#first we need to convert the textual data into the numerical data
#it maintains an internal memory or state 
#rnn does not process the data all at once 
#in rnn we actually have a loop , we feed one word at a time 
#simple rnn layer , it takes the output of previous layer , mix with current input and output us 
#lstm long short term memory , it keep track of all the output at each of the stage
#SENTIMENTAL ANALYSIS
import keras
from keras.datasets import imdb 
from keras.preprocessing import sequence
import tensorflow as tf 
import os 
import numpy as np 
VOCAB_SIZE=88584
MAXLEN=250
BATCH_SIZE=64
(train_data,train_label),(test_data,test_labels)=imdb.load_data(num_words=VOCAB_SIZE)#train_data is simply list of lists of integers
print(train_data[0])#just taking a look
print(len(train_data[0]))
print(len(train_data[1]))
#if we look at the length of each of the train_data we will find all of them different
#this is an issue we cant pass different length into our neural network
'''Therefore, we must make each review the same length. To do this we will follow the procedure below:

if the review is greater than 250 words then trim off the extra words
if the review is less than 250 words add the necessary amount of 0's to make it equal to 250.
Luckily for us keras has a function that can do this for us:'''
train_data=sequence.pad_sequences(train_data,MAXLEN)
test_data=sequence.pad_sequences(test_data,MAXLEN)
#the padding will add to the left side of the text to make it equal to maxlen 
#print(len(train_data[1]))
#print(len(train_data[0]))
#print(train_data[0])
model=tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE,32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
''' It first converts each word in the input sequence to a dense vector using an embedding layer. Then, it uses an LSTM layer to learn long-term dependencies in the sequence. Finally, it uses a dense layer with a sigmoid activation function to produce a single output value which could represent the probability of the input belonging to a specific category.'''
model.summary()

#now its time to compile and train the model 

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['acc'])
history=model.fit(train_data,train_label,epochs=10,validation_split=0.2)
# model's predictions and the true labels (loss)
#optimizer: This specifies the algorithm that updates the model's weights to minimize the chosen loss function. 'rmsprop' is a variant of gradient descent optimizer known for its efficiency.
#metrics: This defines the metrics used to evaluate the model's performance during training and testing. Here, 'acc' (accuracy) is used to measure the percentage of correctly classified reviews.
#model.fit takes four arguements
#validation_split: This specifies the proportion of the training data to be used for validation.
#now we will evaluate the model on our training data to see how well it performs 
results=model.evaluate(test_data,test_labels)
print(results)

# now we want to make the predictions on a movie 
word_index=imdb.get_word_index()
def encode_text(text):
    tokens=keras.preprocessing.text.text_to_word_sequence(text)#this is basically tokenization
    tokens=[word_index[word] if word in word_index else 0 for word in tokens ]
    return sequence.pad_sequences([tokens],MAXLEN)[0]
text='that movie was just amazing , so amazing '
encoded=encode_text(text)
print(encoded)

#decode function 

reverse_word_index={value:key for (key,value) in word_index.items()}
def decode_integers(integers):
    PAD=0
    text=""
    for num in integers:
        if num!=PAD:
            text+=reverse_word_index[num] + " "
    return text[:-1] #all elements of the list except the last one.
print(decode_integers(encoded))

#now we make the predictions 
def predict(text):
    encoded_text=encode_text(text)
    pred=np.zeros((1,250))
    pred[0]=encoded_text
    result=model.predict(pred)
    print(result[0])
positive_review="that movie was so awesome ! i loved it and would watch it again because it was amazingly great"
predict(positive_review)
negative_review='''that movie sucked . I hated it and wouldn't watch it again.was one of the worst things'''
predict(negative_review)
