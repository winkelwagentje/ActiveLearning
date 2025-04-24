import numpy as np
from scipy.optimize import minimize

class ERMinimizer:
    def __init__(self, entry_function, num_rows, num_cols):
        """
        Initializes the ERMinimizer with a function to compute the entries of A
        and the dimensions of A.

        :param entry_function: A function that takes two indices (i, j) and returns the entry A[i, j].
        :param num_rows: Number of rows in matrix A.
        :param num_cols: Number of columns in matrix A.
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.A = self._create_matrix(entry_function)

    def _create_matrix(self, entry_function):
        """
        Creates the matrix A using the provided entry function.

        :param entry_function: Function to compute the entries of A.
        :return: The matrix A.
        """
        return np.array([[entry_function(i, j) for j in range(self.num_cols)] for i in range(self.num_rows)])

    def _f(self, s, q):
        """
        Computes the value of the function f(s, q) = s^T A q.

        :param s: Probability vector s.
        :param q: Probability vector q.
        :return: The value of f(s, q).
        """
        return np.dot(s, np.dot(self.A, q))

    def _max_q_for_fixed_s(self, s):
        """
        Maximizes f(s, q) over q for a fixed s.

        :param s: Fixed probability vector s.
        :return: Maximum value of f(s, q).
        """
        def negative_f(q):
            return -self._f(s, q)

        constraints = {'type': 'eq', 'fun': lambda q: np.sum(q) - 1}
        bounds = [(0, 1) for _ in range(self.num_cols)]
        initial_q = np.full(self.num_cols, 1.0 / self.num_cols)

        result = minimize(negative_f, initial_q, bounds=bounds, constraints=constraints)
        return -result.fun

    def solve(self):
        """
        Solves the minimax problem: min_s max_q f(s, q).

        :return: The optimal probability vector s and the corresponding minimum maximum value.
        """
        def outer_function(s):
            return self._max_q_for_fixed_s(s)

        constraints = {'type': 'eq', 'fun': lambda s: np.sum(s) - 1}
        bounds = [(0, 1) for _ in range(self.num_rows)]
        initial_s = np.full(self.num_rows, 1.0 / self.num_rows)

        result = minimize(outer_function, initial_s, bounds=bounds, constraints=constraints)
        return result.x, result.fun

# Example usage
def entry_function(i, j):
    return i + j  # Example function to generate matrix entries

er_minimizer = ERMinimizer(entry_function, 3, 4)
optimal_s, optimal_value = er_minimizer.solve()

print("Optimal s:", optimal_s)
print("Optimal min_s max_q f(s, q):", optimal_value)