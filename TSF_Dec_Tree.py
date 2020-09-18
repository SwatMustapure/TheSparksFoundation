# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 01:11:31 2020

@author: dell
"""

import os
import numpy as np
import pandas as pd
from pandas import DataFrame , Series
import seaborn as sns
from sklearn import preprocessing 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn import preprocessing
import matplotlib.pyplot as plt

os.chdir('F:')
data=pd.read_csv('Iris_data.csv')
#data preprossesing
print(data.count().sort_values())
df=data
shape=print(df.shape)
df.dropna(how='any')
print(df.head(5))
print(df.dtypes)

df["Species"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)
print(df.head(5))

# to check if there are outliers
from scipy import stats
z=np.abs(stats.zscore(df._get_numeric_data()))
print(z[0:5,:])
df=df[(z<3).all(axis=1)]
shape1=print(df.shape)

if shape1==shape:
    print("There is no outlier in the given data")

import time 
t0=time.time()
x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]      
y=df[["Species"]]

# using the decision tree classifier
from sklearn.externals.six import StringIO 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython.display import SVG
from sklearn.tree import export_graphviz
from graphviz import Source
from IPython.display import Image , display
import pydotplus
from sklearn.metrics import accuracy_score
     
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
clf_dt=DecisionTreeClassifier(random_state=0)

#building the model using training set 
dec_tree_model=clf_dt.fit(x_train,y_train)
print(dec_tree_model)
y_pred=clf_dt.predict(x_test)
score=accuracy_score(y_test,y_pred)
print("Score using Decision tree classifier",score)
print("time taken",time.time()-t0)


import sklearn.metrics as metrics
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn import metrics 
cm=metrics.confusion_matrix(y_test,y_pred)
print(cm)


class_name = ["Iris-setosa","Iris-versicolor","Iris-virginica"]
print(class_name)

#plot of decision tree
graph = Source(tree.export_graphviz(dec_tree_model,out_file=None,feature_names = x_train.columns,class_names = class_name, filled=True))
display(SVG(graph.pipe(format="svg")))






