#!/usr/bin/env python
# coding: utf-8

# 
# THE APPLICATION OF TEXT DATA FOR SENTIMENTAL ANALYSIS
Text Data is  one of the type of data,and can be use for sentimental analysis. The dataset used for 
this project contains label for the emotional content(such as happiness, sadiness and anger) of texts. The dataset contains 1992 rows and 3 columns which are called Unamed, text and sentiment.

# In[1]:



import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import LSTM,Dense, Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

dataset = pd.read_csv('combined_data.csv')

Text Data is  one of the type of data,and can be use for sentimental analysis. The dataset used for 
this project contains label for the emotional content(such as happiness, sadiness and anger) of texts. The dataset contains 1992 rows and 3 columns which are called Unamed, text and sentiment.

# In[2]:


dataset


# In[3]:


dataset.shape


# In[4]:


dataset.isna().sum()

Define the feature and label
# In[5]:


sentences = dataset['text'].tolist()
labels = dataset['sentiment'].tolist()


# In[6]:


#separate the sentences and label in to training and test datasets with ratio of 4 : 1 respectively
training_size = int(len(sentences)* 0.8)

training_sentences = sentences[0: training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]


# In[8]:


# make label into numpy array for use later 
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)


# THE STAGE OF TOKENIZER THE SENTENCES

# In[9]:


vocab_size = 2000
embedding_dim = 10 
max_length = 25
padding_type='post'
truncat_type ='post'
oov_tok ="<OOV>"
# let Tokenizer the training dataset
tokenizer = Tokenizer(num_words=vocab_size, oov_token = oov_tok )
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# make word_index to sequences

sequences = tokenizer.texts_to_sequences(training_sentences)


# In[10]:


# Let pad and truncate the sequence
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncat_type,)
print("Training Padding")
print(padded)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
test_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=truncat_type)
print("Test Padding")
print(test_padded)


# THE STAGE OF EMBEDDING 

# In[11]:


# Define the model
model = Sequential()
model.add(Embedding(vocab_size,embedding_dim, input_length=max_length))
model.add(GlobalMaxPooling1D())
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())


# In[12]:


# compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
num_epochs=12

history = model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(test_padded, testing_labels_final) )

# evaluate the model


# In[15]:


history_dict = history.history
accy = history_dict['accuracy']
val_accy = history_dict['val_accuracy']
epochs = range(1, len(accy) + 1)

pyplot.figure(figsize=(12, 10))
pyplot.plot(epochs, accy, 'g', label = 'Training accuracy')
pyplot.plot(epochs, val_accy, 'b', label = 'Validation accuracy')
pyplot.title('TRAINING AND VALIDATION ACCURACY')
pyplot.xlabel('Epochs')
pyplot.ylabel('Accuracy')
pyplot.legend(loc='upper right')
pyplot.ylim((0.5, 1))
pyplot.show()


# In[ ]:




