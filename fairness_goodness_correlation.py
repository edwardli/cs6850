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

    # each element is an array of arrays that contains the errors for a certain percentage omit
    # rms_errors[0] = [ theirs, just goodness, goodness and bias, exclude] for 0.1 omit
    rms_errors = []
    rms_errors_fg = []
    rms_errors_g = []
    rms_errors_og = []
    for percentage_omit in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        boundary = int(percentage_omit * len(edges))
        test_edges = edges[:boundary]
        known_edges = edges[boundary:]
        rms_error_fg, rms_error_g, rms_error_og = computeRMS(test_edges, known_edges)
        rms_errors_fg.append(rms_error_fg)
        rms_errors_g.append(rms_error_g)
        rms_errors_og.append(rms_error_og)
    omit_values = [x * 10 for x in range(1, 10)]
    line_fg, = plt.plot(omit_values, rms_errors_fg, 'b.-', label='Fairness*Goodness')
    line_g, = plt.plot(omit_values, rms_errors_g, 'k.-', label='Goodness')
    line_og, = plt.plot(omit_values, rms_errors_og, 'y.-', label='OG')
    plt.legend(handles=[line_fg, line_g, line_og])
    plt.xlabel('Percentage of edges removed')
    plt.ylabel('Root Mean Square Error')
    plt.title("BTCAlpha")
    plt.show()

    # Do graphing
    # omit_values = [x * 10 for x in range(1, 10)]
    # f_times_g_rms = []
    # goodness_rms = []
    # goodness_and_bias = []
    # exclude = []
    # for rms_error in rms_errors:
    #     f_times_g_rms.append(rms_error[0])
    #     goodness_rms.append(rms_error[1])
    #     # goodness_and_bias.append(rms_error[2])
    #     # exclude.append(rms_error[3])
    # print(omit_values)
    # print(f_times_g_rms)
    # # plt.plot(omit_values, f_times_g_rms,'b.-', 'hello', omit_values, goodness_rms, 'k.-', 'hello', omit_values,
    # #          goodness_and_bias, 'y.-', 'hello', omit_values, exclude, 'g.-', 'hello',)
    # line_fg, = plt.plot(omit_values, f_times_g_rms, 'b.-', label='Fairness*Goodness')
    # line_g, = plt.plot(omit_values, goodness_rms, 'k.-', label='Goodness')
    # # line_g_bias, = plt.plot(omit_values, goodness_and_bias, 'g.-', label='Goodness+Bias')
    # # line_g_exclude, = plt.plot(omit_values, exclude, 'y.-', label='Exclude')
    # # plt.legend(handles=[line_fg, line_g, line_g_bias, line_g_exclude])
    # plt.legend(handles=[line_fg, line_g])
    # plt.xlabel('Percentage of edges removed')
    # plt.ylabel('Root Mean Square Error')
    # plt.title("BTCAlpha")
    # plt.show()

    # plt.hist(average_errors, 50)
    # plt.show()
    # rms_errors = np.array(rms_errors)
    # rms_means = np.mean(rms_errors,axis=0)
    # print rms_errors
    # print rms_means
    # print rms_errors.shape


def computeRMS(test_edges, known_edges):
    rms_error_fg = []
    rms_error_g = []
    rms_error_og = []
    G = nx.DiGraph()
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G, 1)
    fairness_us, goodness_us = compute_fairness_goodness_us(G, 1)
    fairnesses = []
    goodnesses = []
    actual_w = []

    # Compute actual fairness/goodness
    squared_error = 0.
    error = 0.
    n = 0.

    for u, v, w in test_edges:
        if u in fairness_us and v in goodness_us:
            predicted_w = fairness_us[u] * goodness_us[v]
            squared_error += (w - predicted_w) ** 2
            error += abs(w - predicted_w)
            n += 1

    if n == 0:  # Every edge in test_edges has only 1 edge connected to the graph and it was removed
        return None
    #print "Our oWN approach RMS error 1: %f" % math.sqrt(squared_error / n)
    #print "Our oWN approach Absolute mean error: %f" % (error / n)
    rms_error_og.append(math.sqrt(squared_error / n))

    squared_error = 0.
    error = 0.
    n = 0.

    for u, v, w in test_edges:
        if u in fairness and v in goodness:
            predicted_w = fairness[u] * goodness[v]
            squared_error += (w - predicted_w) ** 2
            error += abs(w - predicted_w)
            n += 1

    if n == 0:  # Every edge in test_edges has only 1 edge connected to the graph and it was removed
        return None
    print(math.sqrt(squared_error / n))
    # print "F*G Absolute mean error: %f" % (error / n)
    rms_error_fg.append(math.sqrt(squared_error / n))

    squared_error = 0.
    error = 0.
    n = 0.


    for u, v, w in test_edges:
        if u in fairness and v in goodness:
            predicted_w = goodness[v]
            squared_error += (w - predicted_w)**2
            error += abs(w - predicted_w)
            n += 1

    if n==0: #Every edge in test_edges has only 1 edge connected to the graph and it was removed
        return None
    #print "G: RMS error 1: %f" % math.sqrt(squared_error / n)
    #print "G: Aboslute mean error: %f" % (error / n)
    rms_error_g.append(math.sqrt(squared_error / n))
    #
    # plt.hist(actual_w, bins = 10)
    # plt.xlabel('Edge Weight')
    # plt.ylabel('Count')
    # plt.title('Edge Weight Distribution on RFA')
    # plt.show()
    return rms_error_fg, rms_error_g, rms_error_og



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


def compute_fairness_goodness_us(G, coeff=1, maxiter=200, epsilon=1e-4):
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

