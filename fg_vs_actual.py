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
    # each element is an array of arrays that contains the errors for a certain percentage omit
    # rms_errors[0] = [ theirs, just goodness, goodness and bias, exclude] for 0.1 omit
    rms_errors = []
    fg_values = []
    actual_values = []
    for percentage_omit in [0.1]:
        boundary = int(percentage_omit * len(edges))
        test_edges = edges[:boundary]
        known_edges = edges[boundary:]
        fg_values, actual_values = computeRMS(test_edges, known_edges)
    plt.scatter(fg_values, actual_values)
    plt.xlabel("Fairness * Goodness")
    plt.ylabel("Actual")
    plt.title("Evaluation on RFA")
    plt.show()

def computeRMS(test_edges, known_edges):

    G = nx.DiGraph()
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G, 1)
    fairness_og, goodness_og = compute_fairness_goodness_og(G, 1)

    # Compute actual fairness/goodness
    squared_error = 0.
    error = 0.
    n = 0.
    fgs = []
    actuals = []
    for u, v, w in test_edges:
        if u in fairness_og and v in goodness_og:
            predicted_w = fairness_og[u] * goodness_og[v]
            squared_error += (w - predicted_w) ** 2
            error += abs(w - predicted_w)
            n += 1
            fgs.append(predicted_w)
            actuals.append(w)

    return fgs, actuals



def predict(G, u, v, goodness, prediction_type="bias"):
    """
    prediction_type is one of "bias", "goodness", or "exclude".
    """
    out_edges = G.out_edges(u, data="weight")
    if (len(out_edges) == 0
        or prediction_type == "goodness"
        or prediction_type == "exclude" and len(out_edges) <= 5):
        return goodness[v]

    average_error = 0.0
    for _, w, weight in out_edges:
        average_error += weight - goodness[w]

    average_error /= len(out_edges)

    prediction = goodness[v] + average_error
    if prediction < -1:
        return -1
    if prediction > 1:
        return 1

    return prediction


def initialize_scores(G):
    fairness = {}
    goodness = {}

    nodes = G.nodes()
    for node in nodes:
        fairness[node] = 1
        try:
            goodness[node] = G.in_degree(node, weight='weight') * 1.0 / G.in_degree(node)
        except:
            goodness[node] = 0
    return fairness, goodness


def compute_fairness_goodness(G, coeff=1, maxiter=200, epsilon=1e-4):
    fairness, goodness = initialize_scores(G)

    nodes = G.nodes()
    iter = 0
    while iter < maxiter:
        if iter == maxiter:
            print("FUCK")
        df = 0
        dg = 0

        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                #g += fairness[edge[0]] * edge[2]
                g += edge[2]

            try:
                dg += abs(g / len(inedges) - goodness[node])
                goodness[node] = g / len(inedges)
            except:
                pass

        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0.0
            for edge in outedges:
                f += 1.0 - coeff * abs(edge[2] - goodness[edge[1]]) / 2.0
            try:
                df += abs(f / len(outedges) - fairness[node])
                fairness[node] = f / len(outedges)
            except:
                pass

        if df < epsilon and dg < epsilon:
            break

        iter += 1

    # print "Total iterations: %d" % iter

    return fairness, goodness


def compute_fairness_goodness_og(G, coeff=1, maxiter=200, epsilon=1e-4):
    fairness, goodness = initialize_scores(G)

    nodes = G.nodes()
    iter = 0
    while iter < maxiter:
        if iter == maxiter:
            print("FUCK")
        df = 0
        dg = 0

        for node in nodes:
            inedges = G.in_edges(node, data='weight')
            g = 0
            for edge in inedges:
                g += fairness[edge[0]] * edge[2]
                #g += edge[2]

            try:
                dg += abs(g / len(inedges) - goodness[node])
                goodness[node] = g / len(inedges)
            except:
                pass

        for node in nodes:
            outedges = G.out_edges(node, data='weight')
            f = 0.0
            for edge in outedges:
                f += 1.0 - coeff * abs(edge[2] - goodness[edge[1]]) / 2.0
            try:
                df += abs(f / len(outedges) - fairness[node])
                fairness[node] = f / len(outedges)
            except:
                pass

        if df < epsilon and dg < epsilon:
            break

        iter += 1

    # print "Total iterations: %d" % iter

    return fairness, goodness

if __name__ == '__main__':
    main()

