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
mpl.use('TkAgg') # Default backend doesn't work on MacOS for some reason

import math
import networkx as nx
import random
import sys
import matplotlib.pyplot as plt

def main():
    file_name = sys.argv[1]

    test_edges = []
    known_edges = []
    with open(file_name, 'r') as f:
        for l in f:
            ls = l.strip().split(",")
            if int(ls[0]) == 1 and int(ls[1]) == 4:
                test_edges.append((ls[0], ls[1], float(ls[2])))
                print("appended test edge")
            if int(ls[0]) == 1 and int(ls[1]) == 2:
                test_edges.append((ls[0], ls[1], float(ls[2])))
            # if int(ls[0]) == 1 and int(ls[1]) == 3:
            #     test_edges.append((ls[0], ls[1], float(ls[2])))
            else:
                known_edges.append((ls[0], ls[1], float(ls[2])))

    G = nx.DiGraph()
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G, 1)

    squared_error = 0
    error = 0
    n = 0
    predicted_w = 0
    actual_w = 0
    fairness_of_1 = 0
    goodness_of_4 = 0
    for u, v, w in test_edges:
        if u in fairness and v in goodness:
            if (u == "1" or u == 1) and (v == "4" or v == 4):
                fairness_of_1 = fairness[u]
                goodness_of_4 = goodness[v]
                predicted_w = fairness[u] * goodness[v]
                actual_w = w

    print "Predicted weight: %f" % predicted_w
    print "Actual_w: %f" % actual_w
    print("Fairness of node 1: %f" % fairness_of_1)
    print("Goodness of node 4: %f" % goodness_of_4)

def initialize_scores(G):
    fairness = {}
    goodness = {}

    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight')*1.0/G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness

def compute_fairness_goodness(G, coeff=1, maxiter=100, epsilon=1e-6):
    fairness, goodness = initialize_scores(G)

    nodes = G.nodes()
    iter = 0
    while iter < maxiter:
        df = 0
        dg = 0

        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]]*edge[2]

            try:
                dg += abs(g/len(inedges) - goodness[node])
                goodness[node] = g/len(inedges)
            except:
                pass

        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0.0
            for edge in outedges:
                f += 1.0 - coeff * abs(edge[2] - goodness[edge[1]])/2.0
            try:
                df += abs(f/len(outedges) - fairness[node])
                fairness[node] = f/len(outedges)
            except:
                pass

        if df < epsilon and dg < epsilon:
            break

        iter+=1

    print "Total iterations: %d" % iter

    return fairness, goodness


if __name__ == '__main__':
    main()

