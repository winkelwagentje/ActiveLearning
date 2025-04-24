from minimax import minimax_solver
import numpy as np
import random

def random_m_n(m, n):
    # generate random m by n matrix over R where each column sums to 0.
    A = np.zeros((m, n))
    for j in range(n):
        for i in range(m - 1):
            A[i, j] = random.randint(-10, 10)

    for j in range(n):
        A[m-1, j] = -1 * sum(A[i, j] for i in range(m-1))

    return A


while True:
    m, n = 4, 2**4
    A = random_m_n(m, n).T
    print(A.T)
    print(f"Row sums: {np.sum(A.T, axis=1)}")
    w, s = minimax_solver(A)
    print(w, s)
    if w < -0.001:
        print(w, A.T)
        break