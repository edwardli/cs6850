from __future__ import division
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
import pandas as pd
import csv

def main():
    file_name = sys.argv[1]
    #test_type = sys.argv[2]  # "l1" or "per" (for leave out one edge or remove percentage of edges)

    edges = []
    with open(file_name, 'r') as f:
        for l in f:
            ls = l.strip().split(",")
            edges.append((ls[0], ls[1], float(ls[2])))

    # Pick percentage of edges to omit at random
    random.shuffle(edges)
    # each element is an array of arrays that contains the errors for a certain percentage omit
    # rms_errors[0] = [ theirs, just goodness, goodness and bias, exclude] for 0.1 omit

    percentage_omit = 0.20
    boundary = int(percentage_omit * len(edges))
    test_edges = edges[:boundary]
    training_edges = edges[boundary:]

    G_train = nx.DiGraph()
    for u, v, w in training_edges:
        G_train.add_edge(u, v, weight=w)

    G_test = nx.DiGraph()
    for u, v, w in test_edges:
        G_test.add_edge(u, v, weight=w)

    create_test_and_train_csv(G_train, G_test, training_edges, test_edges)



def num_common_out_neighbors(g, i, j):
    return len(set(g.successors(i)).intersection(g.successors(j)))

def num_common_in_neighbors(g, i, j):
    return len(set(g.predecessors(i)).intersection(g.predecessors(j)))

def create_test_and_train_csv(G_train, G_test, training_edges, test_edges):
    with open("training_edges.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerow(['in_degree_u', 'in_degree_v', 'out_degree_u', 'out_degree_v', 'num_common_out', 'num_common_in', 'ratio_1', 'ratio_2', 'ratio_3', 'ratio_4', 'avg_ratings_into_v', 'avg_ratings_out_of_u', 'fairness_u', 'fairness_v', 'goodness_u', 'goodness_v', 'weight'])

        # Process training edges
        i = 0
        fairness, goodness = get_fairness_goodness(training_edges)

        for edge in G_train.edges():
            u,v = edge
            weight = G_train.get_edge_data(u,v)['weight']
            in_degree_u = G_train.in_degree(u)
            in_degree_v = G_train.in_degree(v)
            out_degree_u = G_train.out_degree(u)
            out_degree_v = G_train.out_degree(v)
            num_common_out = num_common_out_neighbors(G_train, u, v)
            num_common_in = num_common_in_neighbors(G_train, u, v)
            ratio_1 = (in_degree_u+1) / (in_degree_v+1)
            ratio_3 = (out_degree_u+1) / (out_degree_v+1)
            ratio_2 = (in_degree_u+1) / (out_degree_u+1)
            ratio_4 = (out_degree_u+1) / (in_degree_v+1)

            edges_into_v = G_train.in_edges(v, data = True)
            avg_weight_into_v = 0.
            num_edges_into_v = len(edges_into_v)
            for a,b,data in edges_into_v:
                avg_weight_into_v += data['weight']
            avg_ratings_into_v = (avg_weight_into_v+1) / (num_edges_into_v+1)


            edges_out_of_u = G_train.out_edges(u, data=True)
            avg_weight_out_of_u = 0.
            num_edges_out_of_u = len(edges_out_of_u)
            for a, b, data in edges_out_of_u:
                avg_weight_out_of_u += data['weight']

            avg_ratings_out_of_u = (avg_weight_out_of_u+1) / (num_edges_out_of_u+1)


            # For each of people u rates, for positive ratings, calculate u - goodness of that node

            positive_offness_u = 0
            negative_offness_u = 0
            # For each of people u rates, for negative ratings calculate u - goodness of that node
            fairness_u = fairness[u]
            fairness_v = fairness[v]
            goodness_u = goodness[u]
            goodness_v = goodness[v]
            result_array = []
            result_array.extend((in_degree_u, in_degree_v, out_degree_u, out_degree_v, num_common_out, num_common_in, ratio_1, ratio_2, ratio_3, ratio_4, avg_ratings_into_v, avg_ratings_out_of_u, fairness_u, fairness_v, goodness_u, goodness_v, weight))
            csvWriter.writerow(result_array)
            print("wrote line %d" % i)
            i+=1


def get_fairness_goodness(known_edges):
    rms_error = []
    G = nx.DiGraph()
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G, 1)

    # squared_error = 0.
    # error = 0.
    # n = 0.
    #
    # for u, v, w in test_edges:
    #     if u in fairness and v in goodness:
    #         predicted_w = fairness[u] * goodness[v]
    #         squared_error += (w - predicted_w) ** 2
    #         error += abs(w - predicted_w)
    #         n += 1
    #
    # if n == 0:  # Every edge in test_edges has only 1 edge connected to the graph and it was removed
    #     return None
    # print "RMS error 1: %f" % math.sqrt(squared_error / n)
    # print "Aboslute mean error: %f" % (error / n)
    # rms_error.append(math.sqrt(squared_error / n))

    # print(sum(fairness.values()) / len(fairness.values()))
    return fairness, goodness


def compute_fairness_goodness(G, coeff=1, maxiter=200, epsilon=1e-6):
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


if __name__ == '__main__':
    main()

