#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imporing Libraries


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[4]:


#import Datasets


# In[9]:


df_fake = pd.read_csv("Fake[1].csv")
df_true = pd.read_csv("True[2].csv")


# In[10]:


df_fake.head()


# In[11]:


df_true.head(5)


# In[13]:


# Inserting a column "class" as target feature


# In[15]:


df_fake["class"] = 0
df_true["class"] = 1


# In[16]:


df_fake.shape, df_true.shape


# In[17]:


# Removing last 10 rows for manual testing
df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    

df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[18]:


df_fake.shape, df_true.shape


# In[19]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[20]:


df_fake_manual_testing.head(10)


# In[21]:


df_true_manual_testing.head(10)


# In[22]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# In[23]:


# Merging True and Fake Dataframes


# In[24]:


df_merge = pd.concat([df_fake, df_true], axis =0)
df_merge.head(10)


# In[25]:


df_merge.columns


# In[26]:


# Removing columns which are not required


# In[29]:


df = df_merge.drop(["title", "subject", "date"], axis = 1)


# In[30]:


df.isnull().sum()


# In[31]:


# Random Shuffling the dataframe


# In[32]:


df = df.sample(frac = 1)


# In[33]:


df.head()


# In[34]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[35]:


df.columns


# In[36]:


df.head()


# In[37]:


# Creating a function to process the texts


# In[39]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('http?://\S+|www\.\S+', '',text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[40]:


df["text"] = df["text"].apply(wordopt)


# In[ ]:


# Defining dependent and independent variables


# In[41]:


x = df["text"]
y = df["class"]


# In[ ]:


# Splitting Training and Testing


# In[42]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[ ]:


# Convert text to vectors


# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[ ]:


# Logistic Regression


# In[44]:


from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[45]:


pred_lr=LR.predict(xv_test)


# In[46]:


LR.score(xv_test, y_test)


# In[47]:


print(classification_report(y_test, pred_lr))


# In[48]:


# Decision Tree Classification


# In[49]:


from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[50]:


pred_dt = DT.predict(xv_test)


# In[51]:


DT.score(xv_test, y_test)


# In[52]:


print(classification_report(y_test, pred_dt))


# In[55]:


# Gradient Boosting Classifier


# In[56]:


from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)


# In[57]:


pred_gbc = GBC.predict(xv_test)


# In[58]:


GBC.score(xv_test, y_test)


# In[59]:


print(classification_report(y_test, pred_gbc))


# In[60]:


# Random Forest Classifier


# In[61]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[62]:


pred_rfc = RFC.predict(xv_test)


# In[63]:


RFC.score(xv_test, y_test)


# In[64]:


print(classification_report(y_test, pred_rfc))


# In[65]:


# Model Testing


# In[66]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]),                                                                                                       output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[69]:


news = str(input())
manual_testing(news)


# In[70]:


news = str(input())
manual_testing(news)


# In[71]:


news = str(input())
manual_testing(news)


# In[72]:


news = str(input())
manual_testing(news)

