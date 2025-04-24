import matplotlib.pyplot as plt
import numpy as np


winning_x, losing_x = np.linspace(start=0, stop=0.5), np.linspace(start=0.5, stop=1)
winning_y, losing_y =  -winning_x + 1, -losing_x + 1
optimal_x, optimal_y = [0], [1]
tie_x, tie_y = [0.5], [0.5]
worst_x, worst_y = [1], [0]

plt.plot(winning_x, winning_y, color='orange', label='Winning strategies')
plt.plot(losing_x, losing_y, color='green', label='Losing strategies')
plt.plot(optimal_x, optimal_y, color='red', label='Optimal strategy', linewidth=2, marker='o', markersize=8)
plt.plot(tie_x, tie_y, color='yellow', label='Tying strategy', linewidth=2, marker='o', markersize=8)
plt.plot(worst_x, worst_y, color='green', linewidth=2, marker='o', markersize=8)
plt.xlabel('s_1')
plt.ylabel('s_2')
plt.title('Vizualization of the standard 1-simplex when we can beat random sampling.')
plt.legend()
plt.show()