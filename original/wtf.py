from projected_subgradient import projected_subgradient
import numpy as np
from active_learner import empirical_risk, sampling_risk_diff
from itertools import product
from unbiasedlinearclassifier import UnbiasedLinearClassifier
from sklearn.metrics import mean_squared_error
from random import uniform, randint


def subgradient(s, x, y, u, model, loss):
    TOL = 1e-6
    nu_val = nu(s, x, y, u, model, loss)
    N = len(u)
    nature_space = np.array(list(product([0, 1], repeat=N)))
    vs = []
    for v in nature_space:
        if abs(nu_val - tau_v(s, x, y, u, v, model, loss)) <= TOL:
            vs.append(v)
    # Convex combination (average) of all the risk vectors corresponding to v that reach the maximum.
    return np.mean([risk_vector(x, y, u, v, model, loss) for v in vs], axis=0)

def tau_v(s, x, y, u, v, model, loss):
    N = len(u)
    return sampling_risk_diff(s1=s, s2=1/N * np.ones(N), x=x, y=y, u=u, v=v, model=model, loss=loss)

def nu(s, x, y, u, model, loss):
    N = len(u)
    nature_space = np.array(list(product([0, 1], repeat=N)))
    return max([tau_v(s, x, y, u, v, model, loss) for v in nature_space])

def risk_vector(x, y, u, v, model, loss) -> np.ndarray:
    """
    Define the risk vector as in the thesis.
    """
    N = len(u)
    return np.array([empirical_risk(model=model, loss=loss, features=x + [u[k]], labels=y + [v[k]])
                     for k in range(N)])

# Define an objective function for nu with respect to s.
def objective_nu(s, x, y, u, model, loss):
    """
    Wrapper objective for projected subgradient:
        returns (nu(s), subgradient) with respect to s.
    """
    return nu(s, x, y, u, model, loss), subgradient(s, x, y, u, model, loss)

if __name__ == '__main__':
    # --- Define or load your problem data ---
    # These variables must be set appropriately for your application.
    # For demonstration, we create dummy values.
    
    M = 6
    x, y = np.array([uniform(10, 100) for _ in range(M)]), np.array([randint(0,1) for _ in range(M)])
    
    N = 2
    u = np.array([uniform(0, 100) for _ in range(N)])
    
    # The model and loss should be defined according to your learning problem.
    model = UnbiasedLinearClassifier()
    loss = mean_squared_error
    s0 = 1/N * np.ones(N)
    
    # Wrap the objective to fix the additional parameters.
    def obj(s):
        return objective_nu(s, x, y, u, model, loss)
    
    # Run the projected subgradient method.
    s_opt, nu_opt, history = projected_subgradient(s0, obj, initial_step=0.5, max_iter=1000, tol=1e-8)
    
    print("Optimal s:", s_opt)
    print("Optimal nu(s):", nu_opt)