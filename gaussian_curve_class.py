#from scipy.stats import multivariate_normal
import numpy as np
import math
#import matplotlib.pyplot as plt
#from matplotlib import cm

class gaussian_curve:
    def __init__(self, mu, sigma, pi): #pi = gaussian weighting factor
        self.mu = mu
        self.sigma = sigma
        self.pi = pi
        self.probabilities = None
        self.responsibilities = None

    def set_mu(self, value):
        self.mu = value
    def get_mu(self):
        return self.mu

    def set_sigma(self, value):
        self.sigma = value
    def update_sigma(self,X,Y):
        self.sigma = self.covar(X,Y,self.responsibilities)

    def set_pi(self, value):
        self.pi = value

    def set_probabilities(self, probabilities):
        self.probabilities = probabilities
    def probabilities_at(self, pos):
        return self.multivariate_gaussian(pos)
    def update_probs(self, pos):
        #transforms data that comes in and back as 2d array to 1d
        Z = self.probabilities_at(pos)
        Z1 = Z[:,0]
        Z2 = Z[:,1]
        Z = np.concatenate((Z1,Z2),axis=0)
        self.set_probabilities(Z)

    def set_responsibilities(self, responsibilities):
        self.responsibilities = responsibilities

    #def update_normal(self):
    #    self.normal = multivariate_normal(self.mu,self.sigma)

    def multivariate_gaussian(self, pos):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        Ex: pos[:,:,0] = X = [1,1,1;2,2,2;3,3,3]
            pos[:,:,1] = Y = [2,3,4;2,3,4;2,3,4]

        """

        n = self.mu.shape[0]
        Sigma_det = np.linalg.det(self.sigma)
        Sigma_inv = np.linalg.inv(self.sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-self.mu, Sigma_inv, pos-self.mu)

        return np.exp(-fac / 2) / N

#Still need to factor in the responsibility
    def covar(self, X, Y, R):
        size = len(X)
        #combine matrices to make a 2d array
        A = np.column_stack((X,Y))
        one = np.ones((size, size))
        #calculating devation values and storing in a
        #R is responsibility array of points for the curve
        #a1 is the column means (X and Y) repeated, e.g.
        #a2 is column vectors (X-mu)*R (Y-mu)*R
        #a is column vectors (X-mu) (Y-mu)
        """
        66 90
        66 90
        66 90
        """
        a1 = np.matmul(one, A)
        a1 = a1*(1/size)
        a = np.subtract(A, a1)
        
        a2 = a
        a2[:,0] *= np.multiply(a2[:,0],R)
        a2[:,1] *= np.multiply(a2[:,1],R)
        #to find deviation score sums of sq matrix, compute a'a
        V = np.matmul(np.transpose(a2), a)
        Nk = np.sum(self.responsibilities)
        V = V * (1/Nk)
        return V
    
    def update_pi(self):
        Nk = np.sum(self.responsibilities)
        N = self.responsibilities.size
        self.pi = Nk/N
    
    def update_mu(self, X, Y):
        Nk = np.sum(self.responsibilities)
        points = np.column_stack((X,Y))
        self.mu =  np.matmul(np.transpose(self.responsibilities), points) / Nk
        

def log_likelihood(size, curve1, curve2):
    log_likelihood = 0
    print("Curve1, mu = %f, %f" % (curve1.mu[0], curve1.mu[1]))
    for i in range(size):
        temp = curve1.pi*curve1.probabilities[i]
        temp += curve2.pi*curve2.probabilities[i]
        #natural log
        log_likelihood += math.log1p(temp)
    return log_likelihood

def iterate(curve1, curve2, X, Y):
    #1 update mu
    curve1.update_mu(X,Y)
    curve2.update_mu(X,Y)
    
    #2 update sigma
    curve1.update_sigma(X,Y)
    curve2.update_sigma(X,Y)
    
    #3 update prob
    X_arrs = np.array_split(X,2)
    Y_arrs = np.array_split(Y,2)
    twoD_X = np.column_stack((X_arrs[0],X_arrs[1]))
    twoD_Y = np.column_stack((Y_arrs[0],Y_arrs[1]))
    data_pos = np.empty(twoD_X.shape + (2,))
    data_pos[:, :, 0] = twoD_X
    data_pos[:, :, 1] = twoD_Y
    
    curve1.update_probs(data_pos)
    curve2.update_probs(data_pos)
    
    #4 update resp
    curve1.set_responsibilities((curve1.pi*curve1.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities))
    curve2.set_responsibilities((curve2.pi*curve2.probabilities)/(curve2.pi*curve2.probabilities+curve1.pi*curve1.probabilities))
    
    #5 update pi
    curve1.update_pi()
    curve2.update_pi()
        
