import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gaussian_curve_class as gcc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

#read data
Y_label = 'radius_mean'
X_label = 'concavity_mean'
df = pd.read_csv('data/breast_cancer.csv', usecols = ['diagnosis', Y_label, X_label])
benign = df.loc[df['diagnosis'] == 'B']
malignant = df.loc[df['diagnosis'] == 'M']

#create random means and sigmas and gaussian weighting factors
numClasses = 2
xMax, xMin, yMax, yMin = df.loc[:,X_label].max(), df.loc[:,X_label].min(), df.loc[:,Y_label].max(), df.loc[:,Y_label].min()
mu_1 = np.array([(xMax-xMin)*3/4+xMin,(yMax-yMin)*1/4+yMin])
mu_2 = np.array([(xMax-xMin)*1/4+xMin,(yMax-yMin)*3/4+yMin])
sigma1 = np.array([[(xMax-xMin)/32,0],[0,(yMax-yMin)/2]])
sigma2 = np.array([[(xMax-xMin)/32,0],[0,(yMax-yMin)/2]])
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
newXs = np.array_split(X,2)
newYs = np.array_split(Y,2)
newX = np.column_stack((newXs[0],newXs[1]))
newY = np.column_stack((newYs[0],newYs[1]))


# Pack X and Y into a single 3-dimensional array
X = newX
Y = newY

data_pos = np.empty(X.shape + (2,))
data_pos[:, :, 0] = X
data_pos[:, :, 1] = Y

#calculate distribution values for points in data
curve1.set_probabilities(curve1.probabilities_at(data_pos))
curve2.set_probabilities(curve2.probabilities_at(data_pos))
#################
# calculating responsibility
####### shape changes when probabilities are set for some reason. Put back getters to fix
Z1 = curve1.probabilities_at(data_pos)
print(Z1.shape)

curve1.set_responsibilities(  (curve1.pi*curve1.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities) )
curve2.set_responsibilities(  (curve2.pi*curve2.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities) )


###################
#Plot scatter points
###plot###
bx = benign.plot(kind='scatter',x = X_label, y = Y_label,color='red', label = 'malignant')
malignant.plot(kind='scatter',x = X_label, y = Y_label, color='blue', ax=bx, label='benign')

bx.set_xlabel(X_label)
bx.set_ylabel(Y_label)

plt.plot(mu_1[0],mu_1[1],'gx')
plt.plot(mu_2[0],mu_2[1],'gx')
###

 

