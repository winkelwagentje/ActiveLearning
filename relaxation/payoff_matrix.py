import numpy as np
from itertools import product

def emp_risk(h, x, y, u, v, loss):
    """
    Compute the empirical risk of h.
    Ensure inputs to the loss function are array-like and correctly handle empty inputs.
    """
    M, N = len(x), len(u)

    # Handle empty x cases safely
    if M == 0:
        loss_x = 0
    else:
        x_reshaped = np.atleast_2d(x).reshape(-1, 1)
        y_pred_x = np.array(h(x_reshaped))  # Predictions for labeled data
        y_true_x = np.array(y)
        loss_x = loss(y_true_x, y_pred_x)

    # Handle empty u cases safely
    if N == 0:
        loss_u = 0
    else:
        u_reshaped = np.atleast_2d(u).reshape(-1, 1)
        y_pred_u = np.array(h(u_reshaped))
        y_true_u = np.array(v)
        loss_u = loss(y_true_u, y_pred_u)

    return (loss_x + loss_u) / max(1, M + N)  # Avoid division by zero

def sampling_risk_diff(s, x, y, u, v, loss, model):
    """
    Computes S(s|v) - S(1/N | v) based on empirical risk differences.
    Ensures proper handling of empty x and y.
    """

    N = len(u)
    if N == 0:
        return 0  # No unlabeled data means no risk difference

    res = 0
    for k in range(N):
        # Handle empty x and y correctly
        X_train = np.atleast_2d(np.append(x, u[k])).reshape(-1, 1) if len(x) > 0 else np.array([[u[k]]])
        Y_train = np.append(y, v[k]) if len(y) > 0 else np.array([v[k]])

        # Fit the model and compute empirical risk
        model.fit(X_train, Y_train)
        res += (s[k] - 1/N) * emp_risk(model.predict, x, y, u, v, loss)

    return res

def payoff_matrix(x, y, u, loss, model, codomain):
    """
    Build the payoff matrix for the active learning setting.
    Ensures robustness when x and y are empty.
    """

    M, N = len(x), len(u)
    if N == 0:
        return np.zeros((0, 0))  # No unlabeled data means no payoff matrix

    Y_N = list(product(codomain, repeat=N))  # Cartesian product of codomain of size N
    basis_vectors = np.eye(N)

    L = np.zeros((N, len(Y_N)))  # Matrix size will be C^N rows and N columns
    for i in range(N):
        for j, Y_tuple in enumerate(Y_N):  # Columns correspond to the length of x (i.e., N)
            L[i, j] = sampling_risk_diff(basis_vectors[i], x, y, u, np.array(Y_tuple), loss, model)

    return L.T
