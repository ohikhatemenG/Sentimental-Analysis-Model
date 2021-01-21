#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
dataset = pd.read_csv('combined_data.csv')


# In[5]:


dataset


# In[6]:


dataset.shape


# In[8]:


dataset.isna().sum()


# In[5]:


import  tensorflow as tf
import string
from matplotlib import pyplot
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, GRU, Dropout
from tensorflow.keras.layers import GlobalMaxPooling1D

dataset = pd.read_csv('combined_data.csv')


sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()

# Separate the sentences and labels into training and test sets
training_size = int(len(sentences)*0.8)

training_sentences = sentences[0: training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0: training_size]
testing_labels = labels[training_size:]

# Make labels into numpy arrays for use with the network later
training_labels_final = np.array(training_labels)
test_labels_final = np.array(testing_labels)

vocab_size=1000
embedding_dim=16
max_length=25
trunc_type='post'
padding_type='post'
oov_tok="<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, )
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print("Trainig Padded")
print(padded)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print("Testing Padded")
print(testing_padded)


print(decode_review(padded[1]))
print(training_sentences[1])

# Define model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
num_epochs=10
history = model.fit(padded, training_labels_final, epochs=num_epochs, verbose = 0)
# evaluation of the model
loss, accuracy = model.evaluate(testing_padded, test_labels_final, verbose = 0)
print('Accuracy : %f' % (accuracy* 100))


# In[ ]:





# In[ ]:




