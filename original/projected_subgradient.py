import numpy as np
from simplex_projection import euclidean_proj_simplex

def projected_subgradient(x0, objective, proj=euclidean_proj_simplex, initial_step=0.5, max_iter=2000, tol=1e-3):
    """
    Perform the projected subgradient descent algorithm with a diminishing step size.

    The step size at iteration k is given by:
        alpha_k = initial_step / (k + 1)

    Parameters
    ----------
    x0 : numpy.ndarray
        Initial point.
    objective : callable
        A function that takes a point x and returns a tuple (f, g) where
        f is the objective value and g is a subgradient at x.
    proj : callable, optional
        Projection function to enforce feasibility (default is euclidean_proj_simplex).
    initial_step : float, optional
        Initial step size (default is 0.5).
    max_iter : int, optional
        Maximum number of iterations (default is 1000).
    tol : float, optional
        Tolerance for convergence based on the subgradient norm (default is 1e-8).

    Returns
    -------
    x : numpy.ndarray
        The estimated minimizer.
    f_opt : float
        The objective function value at the optimal x.
    history : list
        List of objective function values for each iteration.
    """
    x = x0.copy()
    history = []

    for k in range(max_iter):
        if k > 0 and k % 100 == 0:
            print(f"projected subgradient iteration: {k}")
        f_val, subgrad = objective(x)
        history.append(f_val)
        
        # Convergence check: if subgradient is small enough, break.
        if np.linalg.norm(subgrad) < tol:
            break
        
        # Diminishing step size: step size = initial_step / (k + 1)
        step_size = initial_step / (k + 1)
        
        # Update: move in the negative subgradient direction.
        x = x - step_size * subgrad
        
        # Project the updated point back onto the simplex.
        x = proj(x)
    
    # Compute the final objective value at the optimal x.
    f_opt, _ = objective(x)
    return x, f_opt, history

# Define two convex differentiable functions (quadratics) and their gradients.
c1 = np.array([0.3, 0.2, 0.5, 0.0])
c2 = np.array([0.1, 0.4, 0.3, 0.2])

def f1(x):
    """Quadratic function centered at c1."""
    return 0.5 * np.linalg.norm(x - c1)**2

def grad_f1(x):
    """Gradient of f1."""
    return x - c1

def f2(x):
    """Quadratic function centered at c2."""
    return 0.5 * np.linalg.norm(x - c2)**2

def grad_f2(x):
    """Gradient of f2."""
    return x - c2

def objective(x):
    """
    Objective function defined as the pointwise maximum of f1 and f2.
    
    Returns
    -------
    f_val : float
        The maximum value between f1(x) and f2(x).
    subgrad : numpy.ndarray
        A subgradient of the max function computed as the average of gradients
        corresponding to the functions achieving the maximum.
    """
    # Evaluate both functions.
    f1_val = f1(x)
    f2_val = f2(x)
    
    # Compute the maximum value.
    f_val = max(f1_val, f2_val)
    
    # Identify which functions attain the maximum (using a small tolerance).
    tol_val = 1e-8
    grads = []
    if np.abs(f1_val - f_val) < tol_val:
        grads.append(grad_f1(x))
    if np.abs(f2_val - f_val) < tol_val:
        grads.append(grad_f2(x))
    
    # Compute the subgradient as the average of the active gradients.
    subgrad = np.mean(grads, axis=0)
    
    return f_val, subgrad

# Example usage:
if __name__ == '__main__':
    # Starting point (feasible for the simplex: nonnegative and sums to 1).
    x0 = np.array([0.25, 0.25, 0.25, 0.25])
    
    # Run the projected subgradient method with a diminishing step size.
    x_opt, f_opt, history = projected_subgradient(x0, objective, initial_step=0.5, max_iter=1000, tol=1e-8)
    
    print("Optimal x:", x_opt)
    print("Optimal f(x):", f_opt)
    print("Objective history:", history)
