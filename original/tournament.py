import numpy as np 
from itertools import product
from random import uniform, randint
from sklearn.metrics import mean_squared_error
from unbiasedlinearclassifier import UnbiasedLinearClassifier

# --- Core Computation Functions (Same as Before) ---

def empirical_risk(model, loss, features, labels):
    if len(features) != len(labels): 
        raise ValueError('Features and labels should have equal length.')
    return np.mean([loss([model.predict(features[k])], [labels[k]]) for k in range(len(features))])

def sampling_risk_diff(s1, s2, x, y, u, v, model, loss):
    N = len(s1)
    if N != len(s2) or len(x) != len(y) or len(u) != len(v):
        raise ValueError('Input lengths mismatch.')

    res = 0
    for k in range(N):
        features = np.concatenate([x, [u[k]]])
        labels = np.concatenate([y, [v[k]]])
        model.fit(features, labels)
        res += (s1[k] - s2[k]) * empirical_risk(model, loss, features, labels)
    return res

def nu(s, model, loss, x, y, u):
    N = len(u)
    unif = np.ones(N) / N
    nature_space = np.array(list(product([0, 1], repeat=N)))
    return max(sampling_risk_diff(s, unif, x, y, u, v, model, loss) for v in nature_space)

# --- Tournament Logic (Same as Before) ---

def pick_winner(u_pair, x, y, model, loss):
    if len(u_pair) == 1:
        return u_pair[0]
    u_indices = [val[1] for val in u_pair]
    left = nu(np.array([1, 0]), model, loss, x, y, u_indices)
    right = nu(np.array([0, 1]), model, loss, x, y, u_indices)
    return u_pair[0] if left <= right else u_pair[1]

def split_list_into_pairs(l):
    if len(l) % 2 == 1:
        return split_list_into_pairs(l[:-1]) + [[l[-1]]]
    return [[l[2*i], l[2*i + 1]] for i in range(len(l)//2)]

def tournament_round(u, x, y, model, loss):
    pairs = split_list_into_pairs(u)
    return [pick_winner(pair, x, y, model, loss) for pair in pairs]

def run_tournament(u, x, y, model, loss):
    while len(u) > 1:
        u = tournament_round(u, x, y, model, loss)
    return u[0][0]

# --- Data Generation ---

def generate_random_data(M, N):
    x = np.array([uniform(10, 100) for _ in range(M)])
    y = np.array([randint(0, 1) for _ in range(M)])
    u_enum = np.array(list(enumerate([uniform(10, 100) for _ in range(N)])))
    return x, y, u_enum

# --- Single Experiment ---

def run_experiment(M, N, model_cls, loss_fn):
    x, y, u_enum = generate_random_data(M, N)
    model = model_cls()
    loss = loss_fn

    winner_index = int(run_tournament(u_enum, x, y, model, loss))
    best_strategy = np.zeros(N)
    best_strategy[winner_index] = 1

    u_values = np.array([pair[1] for pair in u_enum])
    val = nu(best_strategy, model, loss, x, y, u_values)

    success = val < 0
    return success, (val, x, y, u_values, best_strategy)

# --- Generic Experiment Runner ---

def run_experiments(run_experiment_fn, TRIES=1000, **kwargs):
    win_count = 0
    win_list = []

    for i in range(TRIES):
        success, result = run_experiment_fn(**kwargs)
        if success:
            win_count += 1
            win_list.append((i, ) + result)

        if i % 100 == 0:
            print(f"Trial {i}: Current Wins = {win_count}")

    return win_count, win_list

# --- Main ---

def main():
    config = {
        'M': 10,
        'N': 3,
        'model_cls': UnbiasedLinearClassifier,
        'loss_fn': mean_squared_error
    }
    TRIES = 1000

    win_count, win_list = run_experiments(run_experiment_fn=run_experiment, TRIES=TRIES, **config)
    
    print("\n" + "-"*70)
    print(f"Total Wins: {win_count} out of {TRIES}")
    print(f"Winning Cases: {win_list}")

if __name__ == "__main__":
    main()
