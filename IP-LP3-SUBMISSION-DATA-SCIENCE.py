#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[3]:


import pandas as pd
import numpy as np
fname = input('Enter the path to file dataset: ')


# In[4]:


ds = pd.read_csv(fname)


# In[5]:


ds.head()


# In[6]:


import itertools


# In[7]:


from sklearn.feature_extraction.text import TfidfVectorizer 


# In[8]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


ds.shape


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(ds['text'], ds['label'], test_size=0.33, random_state=42)


# In[12]:


vectorizer = TfidfVectorizer()


# In[13]:


x_train1 = vectorizer.fit_transform(x_train)


# In[14]:


from sklearn.linear_model import PassiveAggressiveClassifier


# In[15]:


clf = PassiveAggressiveClassifier()


# In[16]:


clf.fit(x_train1,y_train)


# In[17]:


x_test1 = vectorizer.transform(x_test)
x_test1


# In[18]:


predicted = clf.predict(x_test1)


# In[19]:


print("Accuracy: ",accuracy_score(predicted,y_test))


# In[ ]:


input("prompt: ")


# In[ ]:




