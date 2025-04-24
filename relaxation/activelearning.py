from minimax import minimax_solver
from payoff_matrix import payoff_matrix
import numpy as np

class ActiveLearner:
    def __init__(self):
        """
        Initialize the ActiveLearner's parameters.
        """
        self.A = None

    def build_matrix(self, x, y, u, loss, model, codomain):
        self.A = payoff_matrix(x, y, u, loss, model, codomain)

    def learn(self, x, y, u, loss, model, codomain):
        self.build_matrix(x, y, u, loss, model, codomain)
        return minimax_solver(self.A)


def main():
    print("This module finds the value of a bimatrix game and with this the mixed strategy that minimizes the empirical risk.")

# Example usage
if __name__ == "__main__":
    main()
