import keras
import tensorflow as tf
import numpy as np
from keras.datasets import reuters
from keras_preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

# Tutoirals Follwed: 
#https://towardsdatascience.com/text-classification-in-keras-part-1-a-simple-reuters-news-classifier-9558d34d01d3
#https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
#https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


print(train_data)
print(train_labels)
print(test_data)
print(test_labels)

print('done')


word_index = reuters.get_word_index()

num_classes = max(train_labels)+1
max_words = 10000

token = Tokenizer(max_words)
train_data = token.sequences_to_matrix(train_data)
test_data = token.sequences_to_matrix(test_data)

train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, train_labels, batch_size=32, epochs=2, verbose=1, validation_split=0.1)
score = model.evaluate(test_data, test_labels, batch_size=32, verbose=1)

print("test acc: ", score[1])




