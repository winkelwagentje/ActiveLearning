import numpy as np
from scipy.optimize import linprog


def minimax_solver(A):
    '''
    
    This function takes a 2D numpy array A.
    It then solves the problem of w = min_y max_x x^tr A y.
    It returns the value of the bimatrix game, w, and the value of y
    that is part of the corresponding Nash equilibrium that results in value w.

    '''

    m, n = A.shape

    # we want to calculate min_x c^tr @ x where we use that
    # c = (1    0_n) and x = (w y)
    c = np.array([1] + [0] * n)

    # inequality constraints:
    # A_ub @ x <= b_ub
    A_ub = np.hstack((-np.ones((m, 1)), A))
    b_ub = np.array([0] * m)

    # equality constraints:
    # A_eq @ x = b_eq
    A_eq = np.array([[0] + [1] * n])
    b_eq = 1

    # perform linear programming:
    result = linprog(
        c,
        A_ub,
        b_ub,
        A_eq,
        b_eq,
        bounds = [(None, None)] + [(0, None)] * n
    )

    # Check if optimization succeeded
    if result.success:
        w = result.fun  # The minimized value of w (game value)
        y = result.x[1:]  # Optimal strategy y (omit the first variable, which is w)
        return w, y
    else:
        raise ValueError(f"Optimization failed: {result.message}")



