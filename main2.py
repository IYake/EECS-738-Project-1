import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gaussian_curve_class as gcc
import math
from matplotlib.patches import Ellipse
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import inspect

#read data
Y_label = 'pH'
X_label = 'volatile acidity'
wine = pd.read_csv('data/winequality-red.csv', usecols = ['quality_chr', Y_label, X_label])
class1 = 'G'
class2 = 'B'
class_feature = 'quality_chr'
good = wine.loc[wine[class_feature] == class1]
bad  = wine.loc[wine[class_feature] == class2]
frames = [good, bad]
df = pd.concat(frames, ignore_index = True)
print(df.shape)


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
X_arrs = np.array_split(X,2)
Y_arrs = np.array_split(Y,2)
twoD_X = np.column_stack((X_arrs[0],X_arrs[1]))
twoD_Y = np.column_stack((Y_arrs[0],Y_arrs[1]))



# Pack X and Y into a single 3-dimensional array
data_pos = np.empty(twoD_X.shape + (2,))
data_pos[:, :, 0] = twoD_X
data_pos[:, :, 1] = twoD_Y

#calculate distribution values for points in data
Z1 = curve1.probabilities_at(data_pos)
Z2 = curve2.probabilities_at(data_pos)

#unpack Z into single dimensional array
Z11 = Z1[:,0]
Z12 = Z1[:,1]
Z1 = np.concatenate((Z11,Z12),axis=0)
curve1.set_probabilities(Z1)

Z21 = Z2[:,0]
Z22 = Z2[:,1]
Z2 = np.concatenate((Z21,Z22),axis=0)
curve2.set_probabilities(Z2)

#################
# calculating responsibility
####### shape changes when probabilities are set for some reason. Put back getters to fix
curve1.set_responsibilities((curve1.pi*curve1.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities))
curve2.set_responsibilities((curve2.pi*curve2.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities))

###################
#Plot scatter points
###plot###

bx = good.plot(kind='scatter',x = X_label, y = Y_label,color='red', label = 'good', s = 2)
bad.plot(kind='scatter',x = X_label, y = Y_label, color='blue', ax=bx, label='bad', s =2)
#

steps = 5
for i in range(steps):
    gcc.plot_curves(i,df,X_label,Y_label,class_feature,class1,class2,curve1,curve2)
    print(gcc.log_likelihood(curve1,curve2))
    gcc.iterate(curve1,curve2,X,Y)

# bx.set_xlabel(X_label)
# bx.set_ylabel(Y_label)

# plt.plot(curve1.mu[0],curve1.mu[1],'gx')
# plt.plot(curve2.mu[0],curve2.mu[1],'gx')
#
# ###
#
# a = plt.subplot()
# plt.scatter(X, Y, color = 'grey',s = 2)
#
#
#
# gcc.iterate(curve1,curve2,X,Y)
# gcc.iterate(curve1,curve2,X,Y)
# gcc.iterate(curve1,curve2,X,Y)
# gcc.iterate(curve1,curve2,X,Y)
# cov1 = curve1.sigma
# cov2 = curve2.sigma
#
#
# lambda1_, v1 = np.linalg.eig(cov1)
# lambda2_, v2 = np.linalg.eig(cov2)
#
# lambda1_ = np.sqrt(lambda1_)
# lambda2_ = np.sqrt(lambda2_)
# # print(lambda1_, lambda2_)
# # w, h = 2 * nstd * np.sqrt(vals)
# width1 = lambda1_[0] * 2 *3
# height1 = lambda1_[1] * 2*3
# width2 = lambda2_[0] * 2*3
# height2 = lambda2_[1] * 2*3
# angle1 = math.degrees(math.acos(v1[0, 0]))
# angle2 = math.degrees(math.acos(v2[0, 0]))
#
#
# e1 = Ellipse(curve1.mu, width1, height1, angle1)
# e2 = Ellipse(curve2.mu, width2, height2, angle2)
# e1.set_facecolor('purple')
# e2.set_facecolor('yellow')
# e1.set_alpha(0.5)
# e2.set_alpha(0.5)
# a.add_artist(e1)
# a.add_artist(e2)

#plt.show()
################# MAXIMIZATION STEP ################################


# bx.plot(curve1.mu[0],curve1.mu[1],'co', markersize =15)
# bx.plot(curve2.mu[0],curve2.mu[1],'co', markersize =15)
