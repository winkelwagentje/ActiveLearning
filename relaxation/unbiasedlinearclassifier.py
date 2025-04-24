import numpy as np


'''

In this file we take a 2-class classifier that consists out of a line through the origin, and the sign of the prediction becomes the class.
We take squared loss.
We assume we have two labelled data points, and two unlabeled data points.

It is easily seen that for y=ax we should pick a = (Cov(x,y) + mu_x * mu_y) / (Var(x) + mu_x ^2) to minimize the squared loss.
Our model now becomes h(x) = sgn(ax).

'''

def step(a):
    """
    Applies thresholding to determine the sign of the input.
    If the input is a scalar, it returns -1 if the value is less than 0, otherwise 1.
    If the input is a NumPy array, it performs the operation element-wise.
    """
    a = np.array(a)  # Ensure input is a NumPy array
    return np.where(a < 0, -1, 1)  # Vectorized thresholding


def cov(x, y):
    '''
    Calculates the population covariance between x and y.
    Assumes x and y are lists or arrays of equal length.
    '''
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / n


class UnbiasedLinearClassifier:
    def __init__(self):
        self.a = 0  # Default to zero

    def fit(self, x, y):
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input x and y cannot be empty.")
        
        variance_x = cov(x, x)  # Compute variance
        if variance_x == 0:
            print("Warning: x has zero variance. Setting a to 0.")
            self.a = 0  # Avoid NaN by setting a to 0
        else:
            self.a = cov(x, y) / variance_x

    def predict(self, x):
        x = np.array(x)
        return self.a * x
