# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 01:04:02 2020

@author: pranshu
"""

import pandas as pd

data = pd.read_csv('iris.csv')
X = pd.DataFrame(data.iloc[:,:-1].values)
y = pd.DataFrame(data.iloc[:,4])

from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X = Sc_X.fit_transform(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_y = LabelEncoder()
y = le_y.fit_transform(y)  
y=y.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3)

oe_y = OneHotEncoder(categorical_features=[0])
y_train = oe_y.fit_transform(y_train).toarray()


import keras
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()

model.add(Dense(4,input_dim = 4, activation = 'sigmoid'))
model.add(Dense(4, activation = 'sigmoid'))
model.add(Dense(3, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=1,epochs=100)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.50)

y_pred1=[]
for i in range(len(y_pred)):
    y_pred1.append(0)
    for j in range(len(y_pred[i])):
        if y_pred[i,j] == 1:
            y_pred1[i]= j


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred1)
