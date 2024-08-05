#!/usr/bin/env python
# coding: utf-8

# ## Full Data Science Project - Titanic Survivals Prediction [Classification]

# Predict survival on the Titanic and get familiar with ML basics
# - https://www.kaggle.com/c/titanic

# ### Import Libraries

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Dataset

# In[16]:


titanic = pd.read_csv('titanic.csv')


# In[17]:


titanic.head()


# In[18]:


titanic.info()


# ### Data Cleaning & Pre-processing

# ##### Dealing with Missing Data

# In[19]:


titanic.isnull().sum()


# In[20]:


sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[21]:


titanic['Age'] = titanic['Age'].fillna(titanic['Age'].mean())


# In[24]:


titanic.Age.isnull().sum()


# In[25]:


sns.heatmap(titanic.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[26]:


titanic.drop('Cabin',axis=1,inplace=True)


# In[27]:


titanic.head()


# In[28]:


titanic.dropna(inplace=True)


# ##### Converting Categorical Features 

# In[29]:


titanic.info()


# In[30]:


sex = pd.get_dummies(titanic['Sex'],drop_first=True)
sex


# In[32]:


embark = pd.get_dummies(titanic['Embarked'])
embark


# In[33]:


titanic.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[34]:


titanic = pd.concat([titanic,sex,embark],axis=1)


# In[35]:


titanic.head()


# ### Exploratory Data Analysis

# In[36]:


sns.set_style('whitegrid')


# In[37]:


sns.countplot(x='Survived',data=titanic,palette='pastel')


# In[38]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=titanic,palette='rainbow')


# In[39]:


titanic['Age'].hist(bins=30,color='darkred',alpha=0.7)


# In[40]:


sns.countplot(x='SibSp',data=titanic)


# In[41]:


titanic['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ### Building Our Model

# ##### Train / Test Split Data

# In[42]:


from sklearn.model_selection import train_test_split


# In[49]:


X = titanic.drop('Survived',axis=1)
y = titanic['Survived']


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ##### Model Training and Predicting

# In[51]:


from sklearn.linear_model import LogisticRegression


# In[52]:


model = LogisticRegression(max_iter=5000)


# In[53]:


model.fit(X_train,y_train)


# In[54]:


predictions = model.predict(X_test)


# In[55]:


predictions


# ##### Model Evaluation

# In[56]:


from sklearn.metrics import classification_report


# In[57]:


classification_report(y_test,predictions)


# In[58]:


from sklearn.metrics import confusion_matrix


# In[59]:


confusion_matrix(y_test,predictions)


# ==========

# # THANK YOU!
