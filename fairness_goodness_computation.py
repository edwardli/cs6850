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

    edges = []
    with open(file_name, 'r') as f:
        for l in f:
            ls = l.strip().split(",")
            edges.append((ls[0], ls[1], float(ls[2])))

    # Pick percentage of edges to omit at random
    random.shuffle(edges)
    rms_errors = []
    for percentage_omit in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        rms_error = []
        boundary = int(percentage_omit*len(edges))
        test_edges = edges[:boundary]
        known_edges = edges[boundary:]

        G = nx.DiGraph()
        for u, v, w in known_edges:
            G.add_edge(u, v, weight=w)

        # these two dictionaries have the required scores
        fairness, goodness = compute_fairness_goodness(G, 1)

        squared_error = 0
        error = 0
        n = 0

        for u, v, w in test_edges:
            if u in fairness and v in goodness:
                predicted_w = fairness[u] * goodness[v]
                squared_error += (w - predicted_w)**2
                error += abs(w - predicted_w)
                n += 1

        print "RMS error 1: %f" % math.sqrt(squared_error / n)
        print "Aboslute mean error: %f" % (error / n)
        rms_error.append(math.sqrt(squared_error / n))

        #print(sum(fairness.values()) / len(fairness.values()))

        for prediction_type in ("goodness", "bias", "exclude"):
            squared_error = 0
            error = 0
            n = 0

            for u, v, w in test_edges:
                if u in fairness and v in goodness:
                    predicted_w = predict(G, u, v, goodness, prediction_type)
                    squared_error += (w - predicted_w) ** 2
                    error += abs(w - predicted_w)
                    n += 1

            print "Prediction type is " + prediction_type
            print "RMS error 2: %f" % math.sqrt(squared_error / n)
            print "Aboslute mean error: %f" % (error / n)
            rms_error.append(math.sqrt(squared_error / n))

        rms_errors.append(rms_error)

        # plt.hist(average_errors, 50)
        # plt.show()

    print rms_errors

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

