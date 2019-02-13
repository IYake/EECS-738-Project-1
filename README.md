# EECS-738-Project-1
Machine Learning project. Build probability mixture models for Kaggle datasets.

Expectation Maximization Algorithm:

Our algorithm consists of 4 steps: 
  1) placing a random mean, finding the standard deviation and the weighting factor for each Gaussian. 
  2) Assigning a responsibility for each point. 
  3) finding new means, standard deviation, and weighting factor taking into account the responsibility
  calculated in the second step. 
  4) The algorithm will now check to see if the log likelihood is less than the stepping factor. If so, the algorithm stops,        if not the algorithm will interate through the second and third step until this condition is met. 

Data Set One:

Our first data set, breast cancer we chose 2 clusters since there is two different type of tumors being described in the data set: benign and malignant.

Data Set Two:


How to Compile: 


References: 

