#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import os
#os.chdir("C:\\Users\manas\Downloads\DS")


# In[2]:


import pandas as pd
import numpy as np
fname = input("Provide the path of the dataset file: ")

# In[3]:


ds = pd.read_csv(fname)


# In[4]:


ds.head()


# In[53]:


import itertools


# In[54]:


from sklearn.feature_extraction.text import TfidfVectorizer 


# In[55]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


ds.shape


# In[58]:


x_train, x_test, y_train, y_test = train_test_split(ds['text'], ds['label'], test_size=0.33, random_state=42)


# In[59]:


vectorizer = TfidfVectorizer()


# In[60]:


x_train1 = vectorizer.fit_transform(x_train)


# In[61]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[62]:


clf = PassiveAggressiveClassifier()


# In[63]:


clf.fit(x_train1,y_train)


# In[64]:


x_test1 = vectorizer.transform(x_test)
x_test1


# In[66]:


predicted = clf.predict(x_test1)


# In[68]:


print("Accuracy Score: ",accuracy_score(predicted,y_test))


# In[69]:


confusion_matrix(y_test,predicted)


# In[ ]:




