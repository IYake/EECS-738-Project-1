# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:59:21 2019

@author: Ian
"""

class gaussian_curve:
    def __init__(self, mean, cov_matrix, pi): #pi = gaussian weighting factor
        self.mean = mean
        self.cov_matrix = cov_matrix
        self.pi = pi
        
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