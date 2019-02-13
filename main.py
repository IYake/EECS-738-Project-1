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
Y = 'radius_mean'
X = 'concavity_mean'

df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', Y, X])
benign = df.loc[df['diagnosis'] == 'B']
malignant = df.loc[df['diagnosis'] == 'M']

#plot data
ax = benign.plot(kind='scatter',x = X, y = Y,color='red', label = 'malignant')
malignant.plot(kind='scatter',x = X, y = Y, color='blue', ax=ax, label='benign')

#create random means and stdevs and gaussian weighting factor
numClasses = 2

xMax, xMin, yMax, yMin = df.loc[:,X].max(), df.loc[:,X].min(), df.loc[:,Y].max(), df.loc[:,Y].min()
mean1 = np.array([(xMax-xMin)*3/4+xMin,(yMax-yMin)*1/4+yMin])
mean2 = np.array([(xMax-xMin)*1/4+xMin,(yMax-yMin)*3/4+yMin])

cov_matrix1 = np.array([[(xMax-xMin)/4,0],[0,(yMax-yMin)/4]])
cov_matrix2 = np.array([[(xMax-xMin)/4,0],[0,(yMax-yMin)/4]])

pi1 = 1.0/numClasses
pi2 = pi1

gcc1 = gcc.gaussian_curve(mean1, cov_matrix1, pi1)
gcc2 = gcc.gaussian_curve(mean2, cov_matrix2, pi1)

#assign responsibility for each point to each gaussian curve
#find new means, stdevs, and gaussian weighting factors









###plot###
#.plot(x='x', y='y', ax=ax, style='gx', label='mean2')
#mean1.plot(x='x', y='y', ax=ax, style='gx', label='mean1')
ax.set_xlabel(X)
ax.set_ylabel(Y)
#continue until likelihood difference between iterations passes tolerance level
gcc = gcc.gaussian_curve(mean1,20,30)

plt.plot(mean1[0],mean1[1],'gx')
plt.plot(mean2[0],mean2[1],'gx')
#plt.plot([1,2,3,4], [1,4,9,16], 'ro')
#plt.axis([0, 6, 0, 20])
#plt.show()

 

