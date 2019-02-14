from scipy.stats import multivariate_normal

class gaussian_curve:
    def __init__(self, mean, cov_matrix, pi): #pi = gaussian weighting factor
        self.mean = mean
        self.cov_matrix = cov_matrix
        self.pi = pi
        self.normal = multivariate_normal(mean,cov_matrix)
        
    def set_mean(self, value):
        self.mean = value
    def get_mean(self):
        return self.mean
    
    def set_cov_matrix(self, value):
        self.cov_matrix = value
    def get_cov_matrix(self):
        return self.cov_matrix

    def set_pi(self, value):
        self.pi = value
    def get_pi(self):
        return self.pi
    
    def update_normal(self):
        self.normal = multivariate_normal(self.mean,self.cov_matrix)

#Still need to factor in the responsibility
    def covar(self, X, Y):
        size = len(X)
        #combine matrices to make a 2d array
        A = np.column_stack(X,Y)
        one = np.ones((size, size))
        #calculating devation values and storing in a
        a1 = np.matmul(one, A)
        a1 = a1*(1/size)
        a = np.subtract(A, a1)
        #to find deviation score sums of sq matrix, compute a'a
        V = np.matmul(np.transpose(a), a)
        V = V * (1/size)
        return V
        
