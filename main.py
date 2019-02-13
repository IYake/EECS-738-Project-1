import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gaussian_curve_class as gcc

#read data
Y_label = 'radius_mean'
X_label = 'concavity_mean'

df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', Y_label, X_label])
benign = df.loc[df['diagnosis'] == 'B']
malignant = df.loc[df['diagnosis'] == 'M']

#plot data

#create random means and sigmas and responsibilities
numClasses = 2

xMax, xMin, yMax, yMin = df.loc[:,X_label].max(), df.loc[:,X_label].min(), df.loc[:,Y_label].max(), df.loc[:,Y_label].min()
mu_1 = np.array([(xMax-xMin)*3/4+xMin,(yMax-yMin)*1/4+yMin])
mu_2 = np.array([(xMax-xMin)*1/4+xMin,(yMax-yMin)*3/4+yMin])

sigma1 = np.array([[(xMax-xMin)/4,0],[0,(yMax-yMin)/4]])
sigma2 = np.array([[(xMax-xMin)/4,0],[0,(yMax-yMin)/4]])

pi1 = 1.0/numClasses
pi2 = pi1

gcc1 = gcc.gaussian_curve(mu_1, sigma1, pi1)
gcc2 = gcc.gaussian_curve(mu_2, sigma2, pi1)

#assign responsibility for each point to each gaussian curve
#find new means, stdevs, and gaussian weighting factors


###plot###
ax = benign.plot(kind='scatter',x = X_label, y = Y_label,color='red', label = 'malignant')
malignant.plot(kind='scatter',x = X_label, y = Y_label, color='blue', ax=ax, label='benign')

ax.set_xlabel(X_label)
ax.set_ylabel(Y_label)

gcc = gcc.gaussian_curve(mu_1,20,30)

plt.plot(mu_1[0],mu_1[1],'gx')
plt.plot(mu_2[0],mu_2[1],'gx')

 

