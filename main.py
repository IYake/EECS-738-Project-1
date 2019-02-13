# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:45:21 2019

@author: Ian
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gaussian_curve_class as gcc

#read data
df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', 'radius_mean', 'fractal_dimension_worst'])
benign = df.loc[df['diagnosis'] == 'B']
malignant = df.loc[df['diagnosis'] == 'M']

#plot data
ax = benign.plot(kind='scatter',x = 'fractal_dimension_worst', y ='radius_mean',color='red', label = 'malignant')
malignant.plot(kind='scatter',x = 'fractal_dimension_worst', y ='radius_mean', color='blue', ax=ax, label='benign')


#create random means and stdevs and gaussian weighting factor
numClasses = 2

mean1 = np.array([0.16,20])
mean2 = np.array([0.06,10])

cov_matrix1 = np.array([[0.02,0],[0,5]])
cov_matrix2 = np.array([[0.02,0],[0,5]])

pi1 = 1.0/numClasses
pi2 = pi1

gcc1 = gcc.gaussian_curve(mean1, stdev1, pi1)
gcc2 = gcc.gaussian_curve(mean2, stdev2, pi1)

#assign responsibility for each point to each gaussian curve
#find new means, stdevs, and gaussian weighting factors

###plot###

#.plot(x='x', y='y', ax=ax, style='gx', label='mean2')
#mean1.plot(x='x', y='y', ax=ax, style='gx', label='mean1')
ax.set_xlabel("fractal_dimension_worst")
ax.set_ylabel("radius_mean")
#continue until likelihood difference between iterations passes tolerance level
gcc = gcc.gaussian_curve(mean1,20,30)

plt.plot(mean1[0],mean1[1],'gx')
#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt.axis([0, 6, 0, 20])
#plt.show()

 

