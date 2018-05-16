import sys
import matplotlib.pyplot as plt
import math



def plot_graph():
    file_name = sys.argv[1]
    #test_type = sys.argv[2]  # "l1" or "per" (for leave out one edge or remove percentage of edges)

    omit_values = [x * 10 for x in range(1, 10)]
    ml_rmse = []
    fg_rmse = []
    with open(file_name, 'r') as f:
        for l in f:
            ls = l.strip().split(",")
            ml_rmse.append(ls[0])
            fg_rmse.append(ls[1])

    omit_values = [x * 10 for x in range(1, 10)]
    line_fg, = plt.plot(omit_values, fg_rmse, 'b.-', label='Fairness*Goodness')
    line_ml, = plt.plot(omit_values, ml_rmse, 'k.-', label='Linear Regression')
    plt.legend(handles=[line_fg, line_ml])
    plt.xlabel('Percentage of edges removed')
    plt.ylabel('Root Mean Square Error')
    plt.title("Linear Regression on BTC OTC")
    plt.show()

    # Pick percentage of edges to omit at random

if __name__ == '__main__':
    plot_graph()