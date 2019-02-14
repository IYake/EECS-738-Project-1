import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gaussian_curve_class as gcc
from matplotlib import cm


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

curve1 = gcc.gaussian_curve(mu_1, sigma1, pi1)
curve2 = gcc.gaussian_curve(mu_2, sigma2, pi1)

#assign responsibility for each point to each gaussian curve
#find new means, stdevs, and gaussian weighting factors


###plot###
"""ax = benign.plot(kind='scatter',x = X_label, y = Y_label,color='red', label = 'malignant')
malignant.plot(kind='scatter',x = X_label, y = Y_label, color='blue', ax=ax, label='benign')

ax.set_xlabel(X_label)
ax.set_ylabel(Y_label)

plt.plot(mu_1[0],mu_1[1],'gx')
plt.plot(mu_2[0],mu_2[1],'gx')"""

#N = 60
#X = np.linspace(xMin/2, xMax*4/3, N)
#Y = np.linspace(yMin/2, yMax*4/3, N)
#X, Y = np.meshgrid(X, Y)
numPoints = df.shape[0]
if (numPoints % 2 != 0):
    X = df.loc[:numPoints-2,X_label].values
    Y = df.loc[:numPoints-2,Y_label].values
else:
    X = df.loc[:numPoints,X_label].values
    Y = df.loc[:numPoints,Y_label].values

#X = df.loc[1:568,X_label].values

print("X dimension: %i" % X.shape[0])
newXs = np.array_split(X,2)
newYs = np.array_split(Y,2)

newX = np.column_stack((newXs[0],newXs[1]))
newY = np.column_stack((newYs[0],newYs[1]))
#print(newX)
# Mean vector and covariance matrix
#mu = np.array([0., 1.])
#Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])
#curve1.set_mu(mu)
#curve1.set_sigma(Sigma)

# Pack X and Y into a single 3-dimensional array
X = newX
Y = newY
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y

Z = curve1.multivariate_gaussian(pos)
###################
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True,
               #cmap=cm.viridis)

cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(50, -120)

plt.show()
 

