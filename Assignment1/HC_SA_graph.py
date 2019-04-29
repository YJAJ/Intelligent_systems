import numpy as np
import matplotlib.pyplot as plt

def graph_steps_hcost(heuristic_costs, step):
    # for HC and SA
    x = np.array([i for i in range(1,step+1)])
    y = np.array(heuristic_costs)

    # graph scatter dots with labels and legends
    plt.scatter(x, y, s=10)
    plt.xlabel('Steps taken')
    plt.ylabel('Heuristic cost')
    plt.show()