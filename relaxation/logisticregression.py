from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import log_loss
from activelearning import ActiveLearner


codomain = [0, 1]
loss = lambda x, y: log_loss(x, y, labels=codomain)
model = LogisticRegression()
# note that y must contain both 0 and 1 otherwise python mad.
x, y, u = np.array([[-100], [6], [12], [100]]), np.array([0, 1, 1, 1]), np.array([-4, -2, 0, 2, 4])

active_learner = ActiveLearner()
value, strategy = active_learner.learn(x, y, u, loss, model, codomain)
print(value, strategy)

payoff = active_learner.A.T 
print(payoff)
