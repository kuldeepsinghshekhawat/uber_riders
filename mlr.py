# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:47:23 2020

@author: kuldeep Singh Shekhawat
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
data=pd.read_csv("taxi.csv")
data.head()
features=data.iloc[:,:-1].values
labels=data.iloc[:,-1].values
print(labels)


#train_test_split
from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.30,random_state=0)


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(features_train,labels_train)
print("Train Score",reg.score(features_train,labels_train))
print("Train Score",reg.score(features_test,labels_test))


#create a model
pickle.dump(reg,open('taxi.pickle','wb'))
model=pickle.load(open('taxi.pickle','rb'))
print(model.predict([[80,1770000,6000,85]]))








