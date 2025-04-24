from projected_subgradient import projected_subgradient
import numpy as np
from active_learner import empirical_risk, sampling_risk_diff
from itertools import product
from unbiasedlinearclassifier import UnbiasedLinearClassifier
from sklearn.metrics import mean_squared_error
from random import uniform, randint, choice
from functools import reduce
from joblib import Parallel, delayed
import time
import csv

# Toggle detailed logging
VERBOSE = True

# --- Cache nature-space per N ---
_nature_space_cache = {}
def get_nature_space(N):
    if N not in _nature_space_cache:
        _nature_space_cache[N] = np.array(list(product([0, 1], repeat=N)))
    return _nature_space_cache[N]

# --- Core Functions with Parallelization ---

def tau_v(s, x, y, u, v, model, loss):
    return sampling_risk_diff(s1=s,
                              s2=np.ones(len(u)) / len(u),
                              x=x, y=y, u=u, v=v,
                              model=model, loss=loss)

def compute_taus_and_space(s, x, y, u, model, loss):
    nature_space = get_nature_space(len(u))
    taus = Parallel(n_jobs=-1)(
        delayed(tau_v)(s, x, y, u, v, model, loss)
        for v in nature_space
    )
    return np.array(taus), nature_space

def nu(s, x, y, u, model, loss):
    taus, _ = compute_taus_and_space(s, x, y, u, model, loss)
    return np.max(taus)

def risk_vector(x, y, u, v, model, loss):
    return np.array([
        empirical_risk(model=model, loss=loss,
                       features=x + [u[k]], labels=y + [v[k]])
        for k in range(len(u))
    ])

def subgradient(s, x, y, u, model, loss):
    TOL = 1e-6
    taus, nature_space = compute_taus_and_space(s, x, y, u, model, loss)
    nu_val = np.max(taus)
    vs = [v for v, t in zip(nature_space, taus) if abs(nu_val - t) <= TOL]
    grads = Parallel(n_jobs=-1)(
        delayed(risk_vector)(x, y, u, v, model, loss)
        for v in vs
    )
    return np.mean(grads, axis=0)

def objective_nu_factory(x, y, u, model, loss):
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
    return UnbiasedLinearClassifier(), mean_squared_error

# --- Active Learner Strategies ---

def two_point_active_learner(i, j, u, x, y, model, loss):
    if nu(np.array([1, 0]), x, y, [u[i], u[j]], model, loss) < 0:
        return i
    if nu(np.array([0, 1]), x, y, [u[i], u[j]], model, loss) < 0:
        return j
    return choice([i, j])

def winning_index(ids, x, y, u, model, loss):
    return reduce(lambda f, s: two_point_active_learner(f, s, u, x, y, model, loss),
                  ids)

def linear_greedy(x, y, u, model, loss):
    i = winning_index(list(range(len(u))), x, y, u, model, loss)
    s = np.zeros(len(u)); s[i] = 1
    return s

def two_point_order(i, j, x, y, u, model, loss):
    return nu(np.array([1, 0]), x, y, [u[i], u[j]], model, loss) < 0

def NPointOrder(x, y, u, model, loss):
    indices = list(range(len(u)))
    order, equalities = [], []
    for _ in indices.copy():
        best = winning_index(indices, x, y, u, model, loss)
        order.append(best); indices.remove(best)
        if len(order) > 1:
            equalities.append(not two_point_order(order[-2], order[-1], x, y, u, model, loss))
    return order, equalities

def exponentiated_decay(order, equalities, q):
    s_vals = []
    curr = q
    for i in range(len(order)):
        if i > 0 and not equalities[i-1]:
            curr *= q
        s_vals.append(curr)
    Z = sum(s_vals)
    out = np.zeros(len(order))
    for idx, val in zip(order, s_vals):
        out[idx] = val / Z
    return out

def quadratic_greedy(u, x, y, model, loss, q):
    order, equalities = NPointOrder(x, y, u, model, loss)
    return exponentiated_decay(order, equalities, q)

# --- Single-experiment worker ---

def one_experiment(arg):
    M, N, q = arg
    x, y, u = generate_dummy_data(M=M, N=N)
    model, loss = setup_model_and_loss()

    s1 = linear_greedy(x, y, u, model, loss)
    s2 = quadratic_greedy(u, x, y, model, loss, q)
    s1val = nu(s1, x, y, u, model, loss)
    s2val = nu(s2, x, y, u, model, loss)

    s_opt, nu_opt, _ = projected_subgradient(
        np.ones(N)/N,
        objective_nu_factory(x, y, u, model, loss),
        max_iter=1000, tol=1e-3
    )

    return s1, s1val, s2, s2val, s_opt, nu_opt

# --- Entry Point ---

if __name__ == '__main__':
    ITER, N, M, q = 1000, 3, 10, 0.5
    start_outer = time.time()

    if VERBOSE:
        results = []
        for i in range(ITER):
            print(f"Loop iteration: {i}")
            s1, s1val, s2, s2val, s_opt, nu_opt = one_experiment((M, N, q))
            print(f"  linear      -> {s1}, ν = {s1val:.4f}")
            print(f"  quadratic   -> {s2}, ν = {s2val:.4f}")
            print(f"  subgradient -> {s_opt}, ν = {nu_opt:.4f}")
            results.append((s1val, s2val, nu_opt, s1, s2, s_opt))
    else:
        args = [(M, N, q)] * ITER
        raw = Parallel(n_jobs=-1)(
            delayed(one_experiment)(arg) for arg in args
        )
        results = [(s1v, s2v, nuv, s1, s2, so) for (s1, s1v, s2, s2v, so, nuv) in raw]

    total_time = time.time() - start_outer
    print(f"Total time: {total_time:.2f}s")

    with open(f"results_q{q}.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Linear Greedy Val',
            'Quadratic Greedy Val',
            'Projected Subgradient Val',
            'Linear Greedy Strategy',
            'Quadratic Greedy Strategy',
            'Projected Subgradient Strategy'
        ])
        writer.writerows(results)
