"""
Code for the paper:
Edge Weight Prediction in Weighted Signed Networks.
Conference: ICDM 2016
Authors: Srijan Kumar, Francesca Spezzano, VS Subrahmanian and Christos Faloutsos

Author of code: Srijan Kumar
Email of code author: srijan@cs.stanford.edu

Modifications by: Edward Li, John Hughes, James Chen
"""
import matplotlib as mpl

mpl.use('TkAgg')  # Default backend doesn't work on MacOS for some reason

import numpy as np
import math
import networkx as nx
import random
import sys
import matplotlib.pyplot as plt


def main():
    file_name = sys.argv[1]

    edges = []
    with open(file_name, 'r') as f:
        for l in f:
            ls = l.strip().split(",")
            edges.append((ls[0], ls[1], float(ls[2])))

    # Pick percentage of edges to omit at random
    random.shuffle(edges)
    graph_degree_distribution(edges)


def graph_degree_distribution(known_edges):
    G = nx.DiGraph()
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)
    in_degrees = []
    out_degrees = []
    max_in_degree = 0
    for u in G.node():
        if G.in_degree(u) > max_in_degree:
            max_in_degree = G.in_degree(u)
        in_degrees.append(G.in_degree(u))
        out_degrees.append(G.out_degree(u))
    print(max_in_degree)
    plt.hist(in_degrees, range = (0, 50), bins = [x*2 for x in range(0, 25)])
    plt.xlabel("In-degree")
    plt.ylabel("Count")
    plt.title("In-degree distribution of RFA")
    plt.xticks(np.arange(0, 51, 5.0))
    plt.show()


if __name__ == '__main__':
    main()

