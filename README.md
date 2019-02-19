# EECS-738-Project-1
Machine Learning project. Build probability mixture models for Kaggle datasets.

# Expectation Maximization Algorithm:

Our algorithm consists of 4 steps: 
  1) Placing a random mean, finding the standard deviation and the weighting factor for each Gaussian. 
  2) Assigning a responsibility for each point. 
  3) Finding new means, standard deviation, and weighting factor taking into account the responsibility
  calculated in the second step. 
  4) The algorithm will now check to see if the log likelihood is less than the stepping factor. If so, the algorithm stops,        if not the algorithm will interate through the second and third step until this condition is met. 

# Data Set One:

Our first data set, breast cancer, we chose to use EM with two clusters since there is two different type of tumors being described in the data set: benign and malignant. We then chose two independent variables (radius mean and concavity mean) to analyize within the data set. 

# Data Set Two:

For our second data set we chose the Wine Quality Dataset. Since the quality of the wine was on a scale from 1-10, we catergorized the wine into good and bad categories. The bad wine category ranged from quality ratings of 1-4 and the good wine category ranged from 7-10. We decided to use EM with two clusters and two independent variables: pH level and volatile acidity of the wine. In our model we are only including values with a quality rating of 1-4 and 7-10. 

# How to Compile: 


# References: 

https://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf

https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

“Mixture Models and EM” Pattern Recognition and Machine Learning, by Christopher M. Bishop, Springer, 2007, pp. 437–439.

https://stackoverflow.com/questions/20126061/creating-a-confidence-ellipses-in-a-sccatterplot-using-matplotlib
