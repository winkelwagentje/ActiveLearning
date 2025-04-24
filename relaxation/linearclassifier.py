from activelearning import ActiveLearner
from sklearn.metrics import mean_squared_error
import numpy as np
from random import randint
from relaxation.unbiasedlinearclassifier import UnbiasedLinearClassifier


codomain = [-1, 1]
loss = mean_squared_error
model = UnbiasedLinearClassifier()

strategies_better_than_random = []

while len(strategies_better_than_random) < 10:
    x, y, u = np.array([[randint(-100, 100)], [randint(-100, 100)]]), np.array([-1, 1]), np.array([randint(-100, 100), randint(-100, 100), randint(-100, 100)])
    active_learner = ActiveLearner()
    value, strategy = active_learner.learn(x, y, u, loss, model, codomain)
    print(value, strategy)
    print(f"x: {x}, y: {y}, u: {u}")
    if value < -0.001:
        temp = [x, y, u, value, strategy]
        strategies_better_than_random.append(temp)

print("NOW THE LIST COMES")
print("-----------------------------------------------------------------")

amt = 0
mixed = []

for l in strategies_better_than_random:
    print(l)
    if not any(np.array_equal(l[3], arr) for arr in [np.array([1, 0]), np.array([0, 1])]):
        mixed.append(l)
        amt += 1

if amt == 0:
    print("NOT FOUND")
else:
    for m in mixed:
        print(m)