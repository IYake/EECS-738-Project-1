# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:45:21 2019

@author: Ian
"""

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', 'radius_mean', 'fractal_dimension_worst'])
benign = df.loc[df['diagnosis'] == 'B']
malignant = df.loc[df['diagnosis'] == 'M']

ax = benign.plot(kind='scatter',x = 'fractal_dimension_worst', y ='radius_mean',color='red', label = 'malignant')
malignant.plot(kind='scatter',x = 'fractal_dimension_worst', y ='radius_mean', color='blue', ax=ax, label='benign')



#create random means and stdevs and gaussian weighting factor
mean1 = pd.DataFrame({'x': [0.16], 'y': [20]})
mean2 = pd.DataFrame({'x': [0.06], 'y': [10]})
#assign responsibility for each point to each gaussian curve
#find new means, stdevs, and gaussian weighting factors

###plot###

mean2.plot(x='x', y='y', ax=ax, style='gx', label='mean2')
mean1.plot(x='x', y='y', ax=ax, style='gx', label='mean1')
#continue until likelihood difference between iterations passes tolerance level


 

