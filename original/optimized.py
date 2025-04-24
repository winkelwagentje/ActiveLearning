from projected_subgradient import projected_subgradient
import numpy as np
from active_learner import empirical_risk, sampling_risk_diff
from itertools import product
from unbiasedlinearclassifier import UnbiasedLinearClassifier
from sklearn.metrics import mean_squared_error
from random import uniform, randint, choice
from functools import reduce
import time
import csv

# --- Cache nature-space per N to avoid rebuilding ---
_nature_space_cache = {}
def get_nature_space(N):
    if N not in _nature_space_cache:
        _nature_space_cache[N] = np.array(list(product([0, 1], repeat=N)))
    return _nature_space_cache[N]

# --- Core helper: compute all taus once ---
def compute_taus(s, x, y, u, model, loss):
    N = len(u)
    ns = get_nature_space(N)
    # baseline sampling weights
    uniform_s2 = np.ones(N) / N
    taus = [
        sampling_risk_diff(
            s1=s,
            s2=uniform_s2,
            x=x, y=y, u=u, v=v,
            model=model, loss=loss
        )
        for v in ns
    ]
    return np.array(taus), ns

# --- Core Functions ---

def nu(s, x, y, u, model, loss):
    taus, _ = compute_taus(s, x, y, u, model, loss)
    return np.max(taus)

def subgradient(s, x, y, u, model, loss):
    TOL = 1e-6
    taus, ns = compute_taus(s, x, y, u, model, loss)
    nu_val = np.max(taus)
    # select active v's without recomputing taus
    vs = [v for v, t in zip(ns, taus) if abs(nu_val - t) <= TOL]
    # compute risk vectors
    grads = []
    for v in vs:
        grads.append(risk_vector(x, y, u, v, model, loss))
    return np.mean(grads, axis=0)

def risk_vector(x, y, u, v, model, loss) -> np.ndarray:
    # turn x into a list once
    base = list(x)
    return np.array([
        empirical_risk(
            model=model, loss=loss,
            features=base + [float(u[k])],
            labels=list(y) + [int(v[k])]
        )
        for k in range(len(u))
    ])

def objective_nu_factory(x, y, u, model, loss):
    """
    Returns obj(s) that computes (nu(s), subgradient(s)).
    Both share the same tau computations to avoid repetition.
    """
    def obj(s):
        taus, ns = compute_taus(s, x, y, u, model, loss)
        nu_val = np.max(taus)
        vs = [v for v, t in zip(ns, taus) if abs(nu_val - t) <= 1e-6]
        grads = [risk_vector(x, y, u, v, model, loss) for v in vs]
        return nu_val, np.mean(grads, axis=0)
    return obj

# --- Utility Functions ---

def generate_dummy_data(M=10, N=2):
    x = np.array([uniform(10, 100) for _ in range(M)])
    y = np.array([randint(0, 1) for _ in range(M)])
    u = np.array([uniform(0, 100) for _ in range(N)])
    return x, y, u

def setup_model_and_loss():
    return UnbiasedLinearClassifier(), mean_squared_error

# --- Main Runner ---

def run_optimization(M=3, N=2, initial_step=0.5, max_iter=1000, tol=1e-2):
    x, y, u = generate_dummy_data(M, N)
    model, loss = setup_model_and_loss()
    s0 = np.ones(N) / N

    obj = objective_nu_factory(x, y, u, model, loss)
    s_opt, nu_opt, history = projected_subgradient(
        s0, obj,
        initial_step=initial_step,
        max_iter=max_iter,
        tol=tol
    )
    print_results(s_opt, nu_opt, history)

def print_results(s_opt, nu_opt, history):
    print("Optimal s:", s_opt)
    print("Optimal nu(s):", nu_opt)
    print("Objective history (last 10 values):", history[-10:])

# --- Active Learner Helpers ---

def two_point_active_learner(i, j, u, x, y, model, loss):
    if nu(np.array([1, 0]), x, y, [u[i], u[j]], model, loss) < 0:
        return i
    if nu(np.array([0, 1]), x, y, [u[i], u[j]], model, loss) < 0:
        return j
    return choice([i, j])

def winning_index(ids, x, y, u, model, loss):
    return reduce(
        lambda f, s: two_point_active_learner(f, s, u, x, y, model, loss),
        ids
    )

def linear_greedy(x, y, u, model, loss):
    i = winning_index(list(range(len(u))), x, y, u, model, loss)
    s = np.zeros(len(u)); s[i] = 1
    return s

def two_point_order(i, j, x, y, u, model, loss):
    return nu(np.array([1, 0]), x, y, [u[i], u[j]], model, loss) < 0

def NPointOrder(x, y, u, model, loss):
    indices = list(range(len(u)))
    order, equalities = [], []
    for _ in range(len(u)):
        best = winning_index(indices, x, y, u, model, loss)
        order.append(best)
        indices.remove(best)
        if len(order) > 1:
            equalities.append(
                not two_point_order(order[-2], order[-1], x, y, u, model, loss)
            )
    return order, equalities

def exponentiated_decay(order, equalities, q):
    curr = q
    s_list = [curr]
    for i in range(1, len(order)):
        if not equalities[i - 1]:
            curr *= q
        s_list.append(curr)
    paired = zip(order, s_list)
    sorted_vals = [val for _, val in sorted(paired, key=lambda x: x[0])]
    Z = sum(sorted_vals)
    return np.array(sorted_vals) / Z

def quadratic_greedy(u, x, y, model, loss, q):
    order, equalities = NPointOrder(x, y, u, model, loss)
    return exponentiated_decay(order, equalities, q)

# --- Entry Point ---

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
        print(f"linear {s1}, ν={s1val:.4f}")
        print(f"quadratic {s2}, ν={s2val:.4f}")

        obj = objective_nu_factory(x, y, u, model, loss)
        s_opt, nu_opt, history = projected_subgradient(
            np.ones(N) / N, obj, max_iter=1000, tol=1e-3
        )
        print(f"projected subgradient {s_opt}, ν={nu_opt:.4f}")
        results.append((s1val, s2val, nu_opt, s1, s2, s_opt))

        print(f"  iteration time: {time.time() - start:.2f}s")

    print(f"Total time: {time.time() - start_outer:.2f}s")

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
