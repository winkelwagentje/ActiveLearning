from projected_subgradient import projected_subgradient
import numpy as np
from active_learner import empirical_risk, sampling_risk_diff
from itertools import product
from unbiasedlinearclassifier import UnbiasedLinearClassifier
from sklearn.metrics import mean_squared_error
from random import uniform, randint
import time
from random import choice
from functools import reduce
import csv

# --- Core Functions ---

def subgradient(s, x, y, u, model, loss):
    TOL = 1e-6
    nu_val = nu(s, x, y, u, model, loss)
    N = len(u)
    nature_space = np.array(list(product([0, 1], repeat=N)))
    # num_samples = min(1000, 2**N)
    # nature_space = np.random.randint(0, 2, size=(num_samples, N))
    vs = [v for v in nature_space if abs(nu_val - tau_v(s, x, y, u, v, model, loss)) <= TOL]
    return np.mean([risk_vector(x, y, u, v, model, loss) for v in vs], axis=0)

def tau_v(s, x, y, u, v, model, loss):
    N = len(u)
    return sampling_risk_diff(s1=s, s2=np.ones(N) / N, x=x, y=y, u=u, v=v, model=model, loss=loss)

def nu(s, x, y, u, model, loss):
    N = len(u)
    nature_space = np.array(list(product([0, 1], repeat=N)))
    return max(tau_v(s, x, y, u, v, model, loss) for v in nature_space)

def risk_vector(x, y, u, v, model, loss) -> np.ndarray:
    N = len(u)
    return np.array([
        empirical_risk(model=model, loss=loss, features=x + [u[k]], labels=y + [v[k]])
        for k in range(N)
    ])

def objective_nu_factory(x, y, u, model, loss):
    """
    Returns a function obj(s) that computes (nu(s), subgradient).
    """
    def obj(s):
        return nu(s, x, y, u, model, loss), subgradient(s, x, y, u, model, loss)
    return obj

# --- Utility Functions ---

def generate_dummy_data(M=10, N=2):
    x = np.array([uniform(10, 100) for _ in range(M)])
    y = np.array([randint(0, 1) for _ in range(M)])
    u = np.array([uniform(0, 100) for _ in range(N)])
    return x, y, u

def setup_model_and_loss():
    model = UnbiasedLinearClassifier()
    loss = mean_squared_error
    return model, loss

# --- Main Runner ---

def run_optimization(M=3, N=2, initial_step=0.5, max_iter=1000, tol=1e-2):
    x, y, u = generate_dummy_data(M, N)
    model, loss = setup_model_and_loss()
    s0 = np.ones(N) / N

    obj = objective_nu_factory(x, y, u, model, loss)
    s_opt, nu_opt, history = projected_subgradient(s0, obj, initial_step=initial_step, max_iter=max_iter, tol=tol)

    print_results(s_opt, nu_opt, history)

def print_results(s_opt, nu_opt, history):
    print("Optimal s:", s_opt)
    print("Optimal nu(s):", nu_opt)
    print("Objective history (last 10 values):", history[-10:])

# --- Entry Point ---


def two_point_active_learner(i, j, u, x, y, model, loss):
    first, second = u[i], u[j]
    if nu(np.array([1, 0]), x, y, [first, second], model, loss) < 0:
        return i 
    elif nu(np.array([0, 1]), x, y, [first, second], model, loss) < 0:
        return j 
    return choice([i, j])


def winning_index(ids, x, y, u, model, loss):
    return reduce(lambda f, s: two_point_active_learner(f, s, u, x, y, model, loss), ids)


def linear_greedy(x, y, u, model, loss):
    ids = list(range(len(u)))
    i = winning_index(ids, x, y, u, model, loss)
    s = np.zeros(len(u))
    s[i] = 1
    return s 


def two_point_order(i, j, x, y, u, model, loss):
    u_pair = [u[i], u[j]]
    if nu(np.array([1, 0]), x, y, u_pair, model, loss) < 0:
        return True 
    return False


def NPointOrder(x, y, u, model, loss):
    N = len(u)
    indices = list(range(N))
    order = []
    equalities = []

    for i in range(N):
        best = winning_index(indices, x, y, u, model, loss)
        order.append(best)
        indices.remove(best)
        if i >= 1:
            equalities.append(not two_point_order(order[-2], order[-1], x, y, u, model, loss))
    return order, equalities


def exponentiated_decay(order, equalities, q):
    curr = q
    s_list = [curr]
    for i in range(1, len(order)):
        if not equalities[i - 1]:
            curr = q * curr 
        s_list.append(curr)
    paired = list(zip(order, s_list))
    sorted_values = [val for _, val in sorted(paired, key=lambda x: x[0])]
    Z = sum(sorted_values)
    return np.array(sorted_values) / Z


def quadratic_greedy(u, x, y, model, loss, q):
    order, equalities = NPointOrder(x, y, u, model, loss)
    return exponentiated_decay(order, equalities, q)


if __name__ == '__main__':
    ITER = 1000
    N, M = 3, 10
    q = 0.5
    start_outer = time.time()
    results = []
    for i in range(ITER):
        print(f"Loop iteration: {i}")
        start = time.time()
        x, y, u = generate_dummy_data(M=M, N=N)
        model, loss = setup_model_and_loss()

        s1 = linear_greedy(x, y, u, model, loss)
        s2 = quadratic_greedy(u, x, y, model, loss, q)
        s1val = nu(s1, x, y, u, model, loss)
        s2val = nu(s2, x, y, u, model, loss)
        print(f"linear {s1, s1val}")
        print(f"quadratic {s2, s2val}")
        obj = objective_nu_factory(x, y, u, model, loss)
        s_opt, nu_opt, history = projected_subgradient(np.ones(N) / N, obj, max_iter=1000, tol=1e-3)
        print(f"projected subgradient {s_opt, nu_opt}")
        results.append((s1val, s2val, nu_opt, s1, s2, s_opt))

        end = time.time()
        print(end - start)
    end_outer = time.time()
    total = end_outer - start_outer
    print(total)

    with open(f"resultsq{q}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(['Linear Greedy Val', 'Quadratic Greedy Val', 'Projected Subgradient Val', 'Linear Greedy Strategy', 'Quadratic Greedy Strategy', 'Projected Subgradient Strategy'])

        for result in results:
            writer.writerow(result)
