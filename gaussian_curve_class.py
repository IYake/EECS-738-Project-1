# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 17:59:21 2019

@author: Ian
"""

class gaussian_curve:
    def __init__(self, mean, stdev, pi): #pi = gaussian weighting factor
        self.mean = mean
        self.stdev = stdev
        self.pi = pi
        
    def set_mean(self, value):
        self.mean = value
    def get_mean(self):
        return self.mean
    
    def set_stdev(self, value):
        self.stdev = value
    def get_stdev(self):
        return self.stdev

    def set_pi(self, value):
        self.pi = value
    def get_pi(self):
        return self.pi