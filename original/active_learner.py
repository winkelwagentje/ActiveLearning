import numpy as np
from scipy.optimize import minimize
from itertools import product
from unbiasedlinearclassifier import UnbiasedLinearClassifier
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def empirical_risk(model, loss, features, labels):
    if len(features) != len(labels): 
        raise Exception('Features and labels should have equal length.')

    # Ensure the prediction is wrapped in an array for consistency with mean_squared_error
    return 1/len(features) * sum([loss(np.array([model.predict(features[k])]), np.array([labels[k]])) for k in range(len(features))])



def sampling_risk_diff(s1, s2, x, y, u, v, model, loss):
    N = len(s1)
    if N != len(s2):
        raise Exception('Strategies should have equal length.')
    if len(x) != len(y):
        raise Exception('Features and labels should have equal length.')
    if len(u) != len(v):
        raise Exception('unlabelled data and its targets should have equal length.')
    
    res = 0
    for k in range(N):
        features, labels = np.concatenate([x, np.array([u[k]])]), np.concatenate([y, np.array([v[k]])])
        model.fit(features, labels)
        res += (s1[k] - s2[k]) * empirical_risk(model, loss, features, labels)
    return res


def objective(s, model, loss, x, y, u):
    N = len(u)
    unif = 1/ N * np.ones(N)
    nature_space = np.array(list(product([0, 1], repeat=N)))
    return max(sampling_risk_diff(s, unif, x, y, u, v, model, loss) for v in nature_space)


def active_learning(x, y, u, model, loss):
    N = len(u)
    
    # Define the constraint for the simplex: sum(s) == 1 and s >= 0
    def constraint(s):
        return np.sum(s) - 1  # This ensures the sum of s is 1

    # Initial guess for s: a uniform distribution over the standard simplex
    s0 = np.ones(N) / N
    
    # Bounds for s: each component of s should be between 0 and 1
    bounds = [(0, 1)] * N
    
    # Use scipy.optimize.minimize to minimize the objective function with the constraint
    result = minimize(objective, s0, args=(model, loss, x, y, u), bounds=bounds, constraints={'type': 'eq', 'fun': constraint}, method='SLSQP')
    

    if not result.success:
        raise ValueError("Optimization did not converge: " + result.message)
    
    # The result is the optimal strategy s that minimizes the objective
    return result


def main():
    loss = mean_squared_error
    model = UnbiasedLinearClassifier()

    MAX = 10
    for n in range(1, MAX + 1):
        x, y, u = np.array([0, 1]), np.array([1, 0]), np.array([1.2 + k for k in range(n)])
        result = active_learning(x, y, u, model, loss)
        N = len(u)
        print(f'n: {n}, strategy: {result.x}, objective: {objective(result.x, model, loss, x, y, u)}')


def test():
    loss = mean_squared_error
    model = UnbiasedLinearClassifier()

    MAX = 15
    values = []
    for n in range(1, MAX + 1):
        x, y, u = np.array([0, 1]), np.array([1, 0]), np.array([1.2 + k for k in range(n)])
        N = len(u)
        strat = np.zeros(N)
        strat[N-1] = 1
        value = objective(strat, model, loss, x, y, u)
        values.append(value)
        print(f'Iteration {n} completed, value: {value}')

    plt.plot(range(1, MAX + 1), values)
    plt.xlabel('N')
    plt.ylabel('Value of objective function')
    plt.title('Plot of objective function against N, using the strategy [0, 0, 0, ..., 1]')
    plt.show()


if __name__ == '__main__':
    main()