# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:45:21 2019

@author: Ian
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', 'radius_mean', 'fractal_dimension_worst'])
#print(df)
benign = df.loc[df['diagnosis'] == 'B']
malignant = df.loc[df['diagnosis'] == 'M']
df.plot(kind='scatter',x = 'fractal_dimension_worst', y ='radius_mean', color='red')
 

