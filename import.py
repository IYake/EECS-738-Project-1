# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:45:21 2019

@author: Ian
"""

import pandas as pd

df = pd.read_csv('data/breast_cancer.csv')
radiuses = df.radius_mean
fractals = df.fractal_dimension_worst
diagnoses = df.diagnosis
 

