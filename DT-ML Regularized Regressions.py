#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 12:42:11 2024

@author: hiro
"""

import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
random.seed(42)
import numpy as np
from sklearn.model_selection import KFold, train_test_split


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_boston

import seaborn as sns
#from sklearn.inspection import permutation_importance

os.chdir('/Users/hiro/DS-ML')
from DS_ML_Data import clean
from DS_ML_Data_2 import clean2

import pickle

d, train_df, test_df = clean()


train_df['']

#Model
lr = LinearRegression()

#Fit model
lr.fit(X_train, y_train)

#predict
#prediction = lr.predict(X_test)

#actual
actual = y_test

train_score_lr = lr.score(X_train, y_train)
test_score_lr = lr.score(X_test, y_test)

print("The train score for lr model is {}".format(train_score_lr))
print("The test score for lr model is {}".format(test_score_lr))


#Ridge Regression Model
ridgeReg = Ridge(alpha=10)

ridgeReg.fit(X_train,y_train)

#train and test scorefor ridge regression
train_score_ridge = ridgeReg.score(X_train, y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)

print("\nRidge Model............................................\n")
print("The train score for ridge model is {}".format(train_score_ridge))
print("The test score for ridge model is {}".format(test_score_ridge))