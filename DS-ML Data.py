#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:17:41 2024

@author: hiro
"""


import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics
from statistics import mean
import numpy as np
from scipy.stats import norm    
from functools import reduce  
import operator
from scipy.optimize import minimize    
from numpy import log    
from scipy.interpolate import interp1d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
from torch.optim import Adam

import torch.nn as nn
import torch.optim as optim

os.chdir('/Users/hiro/DS-ML')
df = pd.read_csv('data_30countries.csv')

#dd = pd.read_csv('c13k_selections.csv')

dataframes = {}

for i in range(0,17):
    if i <= 15:
        dataframes[f'df_{i}'] = df.iloc[:, i*10:i*10+9]
    if i == 16:
        dataframes[f'df_{i}'] = df.iloc[:, 160:165]

df0 = dataframes['df_0']

df1 = dataframes['df_1']
df2 = dataframes['df_2']
df3 = dataframes['df_3']
df4 = dataframes['df_4']
df5 = dataframes['df_5']
df6 = dataframes['df_6']
df7 = dataframes['df_7']
df8 = dataframes['df_8']
df9 = dataframes['df_9']
df10 = dataframes['df_10']
df11 = dataframes['df_11']
df12 = dataframes['df_12']
df13 = dataframes['df_13']
df14 = dataframes['df_14']
df15 = dataframes['df_15']
df16 = dataframes['df_16']

kf = KFold(n_splits=8, shuffle=True, random_state=42)

#testing = df0[df0['equiv_nr'] == 3].copy()
#training = df0[df0['equiv_nr'] != 3].copy()

d = df0[["subject_global", "high", "low", "probability", "equivalent", "equiv_nr"]]

d.loc[d['equiv_nr'] == 44, 'low'] = d['equivalent'].astype(d['low'].dtype)
d.loc[d['equiv_nr'] == 44, 'equivalent'] = 0


from sklearn.model_selection import train_test_split

# Assume df is your DataFrame
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

x = torch.tensor(d['high'], dtype=torch.float32).unsqueeze(1)
y = torch.tensor(d['low'], dtype=torch.float32).unsqueeze(1)
p = torch.tensor(d['probability'], dtype=torch.float32).unsqueeze(1)
ce = torch.tensor(d['equivalent'], dtype=torch.float32).unsqueeze(1)