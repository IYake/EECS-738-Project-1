#from scipy.stats import multivariate_normal
import numpy as np
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

    def set_pi(self, value):
        self.pi = value
    
    def set_probabilities(self, probabilities):
        self.probabilities = probabilities
    def probabilities_at(self, pos):
        return self.multivariate_gaussian(pos)
    
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
    def covar(self, X, Y, Z):
        size = len(X)
        #combine matrices to make a 2d array
        A = np.column_stack(X,Y)
        one = np.ones((size, size))
        #calculating devation values and storing in a
        #a1 is the column means (X and Y) repeated, e.g.
        #a2 is column vectors (X-mu)*Z (Y-mu)*Z
        #a is column vectors (X-mu) (Y-mu)
        """
        66 90
        66 90
        66 90
        """
        a1 = np.matmul(one, A)
        a1 = a1*(1/size)
        a = np.subtract(A, a1)
        
        a2 = np.multiply(a,Z)
        #to find deviation score sums of sq matrix, compute a'a
        V = np.matmul(np.transpose(a2), a)
        V = V * (1/size)
        return V
    
    def update_pi(self):
        """print(self.responsibilities)
        print(np.sum(self.responsibilities))
        print(self.responsibilities.size)"""
        self.pi = np.sum(self.responsibilities)/self.responsibilities.size
