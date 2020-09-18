# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 02:32:05 2020

@author: dell
"""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame , Series
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
import matplotlib.pyplot as plt

os.chdir('D:')
data=pd.read_excel('data TSF.xlsx')

#data decription: 
print(data.count().sort_values())
df=data
shape=print(df.shape)

# To remove any null values
df.dropna(how='any')
print(df.head(5))
print(df.dtypes)

#to check if there are outliers
from scipy import stats
z=np.abs(stats.zscore(df._get_numeric_data()))
print(z[0:5,:])
df=df[(z<3).all(axis=1)]
print(df.head(5))
shape1=print(df.shape)

if shape1==shape:
    print("There is no outlier in the given data")
    
df=data
# To plot the scatter plot 
df.plot(kind='scatter', x='Hours', y="Scores")
plt.show()

print(df.corr())

Hours=df[["Hours"]]
Scores=df[["Scores"]]

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(Hours, Scores, test_size=0.2)

# To build the linear regression: 
reg = linear_model.LinearRegression()
model = reg.fit(x_train,y_train)
print(model)

#predicting test data sets
y_pred=reg.predict(x_test)
print(y_pred)

#visualizing the data set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color ='brown')
plt.title("score vs hours")
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()


# The coeffiecient of X is
print('The x coeff is ', reg.coef_)

# The intercept is 
print('The intercept is' , reg.intercept_)


# To calculate R square
from sklearn.metrics import r2_score
R_squared=r2_score(y_test,y_pred)
print('The R squared value is' ,R_squared)

# To predict the new score when hour is 9.25
pred=reg.predict([[9.25]])
print('The predicted score of the student who studies for 9.25 hours is' ,pred)











