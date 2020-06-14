#!/usr/bin/env python
# coding: utf-8

# # Fake News Model using Sklearn

# In[1]:


# Importing all the necessary libraries

import numpy as np
import pandas as pd
import itertools
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 


# In[2]:


# Opening the news.csv file

df = pd.read_csv('news.csv')
df=pd.DataFrame(df)


# In[3]:


# Checking the contents of the file

df.head()


# In[4]:


# Deleting columns having NaN values

for col in df.columns:
    if 'Unnamed' in col:
        del df[col]


# In[5]:


# Checking is deletion is successfull

df


# In[6]:


#printing first 5 entries in the table/dataframe

print(df.head())


# In[7]:


# Droping all the rows having NaN values { Cleaning of data }

df=df.dropna()


# In[8]:


#deciding predicate and predicator variables

labels = df.label
labels.head()


# In[9]:


#splitting data into train dataset and test dataset

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)


# In[10]:


#vectorization and fitting data 

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)


# In[11]:


#predicting lables of test data using PassiveAgrressiveClassifier

pc = PassiveAggressiveClassifier(max_iter=50)
pc.fit(tfidf_train, y_train)


# In[12]:


# Calculation steps for accuracy and confusion matrix

y_pred = pc.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)


# In[13]:


# Printing Accuracy 

print(f'Accuracy: {score}')
print()


# In[14]:


# Printing confusion matrix

confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

