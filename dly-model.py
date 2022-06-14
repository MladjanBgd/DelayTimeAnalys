# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:11:56 2022

@author: mladjan.jovanovic
"""

import pandas as pd

##load csv's to dataframe
#df = pd.concat(map(pd.read_csv, ['D:/RB/2016.csv', 'D:/RB/2017.csv','D:/RB/2018.csv']), ignore_index=True)
df = pd.concat(map(pd.read_csv, ['D:/RB/2016.csv']), ignore_index=True)

##test data
# df.head()
# df.shape
# df.isnull().values.any()
# df.isnull().sum()

#df=df.drop('Unnamed: 27', axis=1)

##fix FL_DATE
df['FL_DATE']=pd.to_datetime(df.FL_DATE, format='%Y-%m-%d')
df['YEAR']=df['FL_DATE'].dt.year
df['MONTH']=df['FL_DATE'].dt.month
df['DAY']=df['FL_DATE'].dt.day
#df=df.drop('FL_DATE', axis=1)

from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder()
df['OP_CARRIER']=ord_enc.fit_transform(df[['OP_CARRIER']])
df['ORIGIN']=ord_enc.fit_transform(df[['ORIGIN']])!
df['DEST']=ord_enc.fit_transform(df[['DEST']])


##make subset
df=df[['YEAR','MONTH', 'DAY', 'OP_CARRIER','ORIGIN','DEST','ARR_DELAY','DISTANCE']]

df=df.head(50000)

#df=df[['YEAR','MONTH', 'DAY', 'ARR_DELAY','DISTANCE']]

#df.isnull().sum()
#df[df.isnull().values.any(axis=1)].head()

##drop NaN
df=df.dropna()

# df.isnull().values.any()

##split dataset to train and test
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DELAY', axis=1), df['ARR_DELAY'], test_size=0.2, random_state=42)

##train a model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=13, warm_start=True)
model.fit(train_x, train_y)

##predict values and score the prediction's
predicted = model.predict(test_x)
print(model.score(test_x, test_y))

##predict prob's
from sklearn.metrics import roc_auc_score
probabilities = model.predict_proba(test_x)

##confusion/errir matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(test_y, predicted))

##
from sklearn.metrics import precision_score
train_predictions = model.predict(train_x)
print(precision_score(train_y, train_predictions, average='weighted'))

##
from sklearn.metrics import recall_score
print(recall_score(train_y, train_predictions, average='weighted'))