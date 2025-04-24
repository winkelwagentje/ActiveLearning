from minimax import minimax_solver
import numpy as np
from sklearn.metrics import mean_squared_error


def main():
    def cov(x, y):
        '''
        Calculates the population covariance between x and y.
        Assumes x and y are lists or arrays of equal length.
        '''
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        return sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / n
    
    def mean(x):
        return sum(x)/len(x)
    
    def a(x, y):
        return (cov(x, y) + mean(x) * mean(y)), (cov(x, x) + mean(x) ** 2)
    
    def emp_risk(h, x, y, u, v, loss):
        """
        Compute the empirical risk of h.
        Ensure inputs to the loss function are array-like.
        """
        M, N = len(x), len(u)

        # Predictions for labeled data
        y_pred_x = np.array(h(x))  # Predictions for labeled data
        y_true_x = np.array(y)

        # Predictions for unlabeled data
        y_pred_u = np.array(h(u))  # Apply the function directly to u
        y_true_u = np.array(v)

        return 1 / (M + N) * (loss(y_true_x, y_pred_x) + loss(y_true_u, y_pred_u))

    
    def step(a):
        """
        Applies thresholding to determine the sign of the input.
        If the input is a scalar, it returns -1 if the value is less than 0, otherwise 1.
        If the input is a NumPy array, it performs the operation element-wise.
        """
        a = np.array(a)  # Ensure input is a NumPy array
        return np.where(a < 0, -1, 1)  # Vectorized thresholding
    
    x, y = np.array([-4, -2, 0, 2, 4]), np.array([1, 1, 1, 1, 1])
    u, v = np.array([-1, 1]), np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    print(f"Value for a: {a(x, y)}")
    print(f"Emp risk: {emp_risk(lambda x: step(-1 * x), x, y, u, v[1], mean_squared_error)}")



if __name__ == "__main__":
    main()