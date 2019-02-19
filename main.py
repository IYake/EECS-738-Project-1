import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import gaussian_curve_class as gcc
import math

from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import inspect
#read data
Y_label = 'radius_mean'
X_label = 'concavity_mean'
class1 = 'B'
class2 = 'M'
class_feature = 'diagnosis'
df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', Y_label, X_label])
df[X_label] = (df[X_label]-df[X_label].mean()) / (df[X_label].max()-df[X_label].min())
df[Y_label] = (df[Y_label]-df[Y_label].mean()) / (df[Y_label].max()-df[Y_label].min())
### create random means and sigmas and gaussian weighting factors ###
numClasses = 2
xMax, xMin, yMax, yMin = df.loc[:,X_label].max(), df.loc[:,X_label].min(), df.loc[:,Y_label].max(), df.loc[:,Y_label].min()
mu_1 = np.array([(xMax-xMin)*3/4+xMin,(yMax-yMin)*1/4+yMin])
mu_2 = np.array([(xMax-xMin)*1/4+xMin,(yMax-yMin)*3/4+yMin])
sigma1 = np.array([[(xMax-xMin)/128,0],[0,(yMax-yMin)/8]])
sigma2 = np.array([[(xMax-xMin)/128,0],[0,(yMax-yMin)/8]])
pi1 = 1.0/numClasses
pi2 = 1.0/numClasses
curve1 = gcc.gaussian_curve(mu_1, sigma1, pi1)
curve2 = gcc.gaussian_curve(mu_2, sigma2, pi1)

#reshape X column and Y column so they can be processed for Z
numPoints = df.shape[0]
if (numPoints % 2 != 0):
    X = df.loc[:numPoints-2,X_label].values
    Y = df.loc[:numPoints-2,Y_label].values
else:
    X = df.loc[:numPoints,X_label].values
    Y = df.loc[:numPoints,Y_label].values
X_arrs = np.array_split(X,2)
Y_arrs = np.array_split(Y,2)
twoD_X = np.column_stack((X_arrs[0],X_arrs[1]))
twoD_Y = np.column_stack((Y_arrs[0],Y_arrs[1]))


# Pack X and Y into a single 3-dimensional array

data_pos = np.empty(twoD_X.shape + (2,))
data_pos[:, :, 0] = twoD_X
data_pos[:, :, 1] = twoD_Y

### calculate distribution values for points in data ###
Z1 = curve1.probabilities_at(data_pos)
Z2 = curve2.probabilities_at(data_pos)

#### set probabilities ###
#unpack Z into single dimensional array
Z11 = Z1[:,0]
Z12 = Z1[:,1]
Z1 = np.concatenate((Z11,Z12),axis=0)
curve1.set_probabilities(Z1)

Z21 = Z2[:,0]
Z22 = Z2[:,1]
Z2 = np.concatenate((Z21,Z22),axis=0)
curve2.set_probabilities(Z2)

### calculating responsibility ###
curve1.set_responsibilities((curve1.pi*curve1.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities))
curve2.set_responsibilities((curve2.pi*curve2.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities))


################# MAXIMIZATION STEP ################################

for i in range(5):
    gcc.plot_curves(i,df,X_label,Y_label,class_feature,class1,class2,curve1,curve2)
    gcc.iterate(curve1,curve2,X,Y)



    
