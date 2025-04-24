import numpy as np

def euclidean_proj_simplex(v, s=1):
    """
    Compute the Euclidean projection of a vector v onto the simplex:
        D_s = { x in R^n : x >= 0, sum(x) = s }.
    
    Parameters
    ----------
    v : numpy.ndarray
        Input vector to be projected.
    s : float, optional
        Sum constraint for the simplex (default is 1).

    Returns
    -------
    w : numpy.ndarray
        The projected vector.
    """
    n = v.shape[0]
    if s < 0:
        raise ValueError("The sum s must be non-negative")
    
    # Sort v in descending order.
    u = np.sort(v)[::-1]
    
    # Compute the cumulative sum of u minus s.
    cssv = np.cumsum(u) - s
    
    # Find the rho value: the largest index where u > (cssv / (i+1)).
    rho = np.nonzero(u > cssv / (np.arange(n) + 1))[0][-1]
    
    # Compute the threshold theta.
    theta = cssv[rho] / (rho + 1.0)
    
    # Compute the projection by thresholding at theta.
    w = np.maximum(v - theta, 0)
    return w

# Example usage:
if __name__ == '__main__':
    # Define an example vector.
    v = np.array([0.2, -0.5, 2.0, 0.1])
    
    # Project the vector onto the unit simplex.
    projected_v = euclidean_proj_simplex(v)
    
    print("Original vector:", v)
    print("Projected vector:", projected_v)
