from relaxation.unbiasedlinearclassifier import UnbiasedLinearClassifier
import matplotlib.pyplot as plt
from activelearning import ActiveLearner
from sklearn.metrics import mean_squared_error
from datetime import datetime


N_max = 20
codomain = [-1, 1]
loss = mean_squared_error
model = UnbiasedLinearClassifier()

for N in range(2, N_max + 1):
    start = datetime.now()
    u = range(2, N + 2)
    al = ActiveLearner()
    value, strat = al.learn(x=[0, 1], y=[1, 1], u=u, loss=loss, model=model, codomain=codomain)
    print(f'n: {N}, value: {value}, strat: {strat}, time taken: {datetime.now() - start}')

