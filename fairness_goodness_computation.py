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

import numpy as np
import math
import networkx as nx
import random
import sys
import matplotlib.pyplot as plt
import pdb

def main():
    file_name = sys.argv[1]
    test_type = sys.argv[2] #"l1" or "per" (for leave out one edge or remove percentage of edges)

    edges = []
    with open(file_name, 'r') as f:
        for l in f:
            ls = l.strip().split(",")
            edges.append((ls[0], ls[1], float(ls[2])))

    # Pick percentage of edges to omit at random
    # each element is an array of arrays that contains the errors for a certain percentage omit
    # rms_errors[0] = [ theirs, just goodness, goodness and bias, exclude] for 0.1 omit
    rms_errors = []

    if test_type == "l1":
        random.shuffle(edges)
        for i in range(50):
            test_edges = [edges.pop(i)]
            rms_error = computeRMS(test_edges, edges)
            if rms_error:
                rms_errors.append(rms_error)
            edges.insert(i,test_edges[0])
    elif test_type == "per":
        for percentage_omit in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            random.shuffle(edges)
            boundary = int(percentage_omit*len(edges))
            test_edges = edges[:boundary]
            known_edges = edges[boundary:]
            print(len(known_edges))
            rms_error = computeRMS(test_edges, known_edges)
            if rms_error:
                rms_errors.append(rms_error)

    rms_errors = np.array(rms_errors)
    rms_means = np.mean(rms_errors,axis=0)
    print rms_errors
    print rms_means
    print rms_errors.shape

    if test_type == "per":
        plotRMSErrors(rms_errors,file_name)

def computeRMS(test_edges, known_edges):
    G = nx.DiGraph()
    for u, v, w in known_edges:
        G.add_edge(u, v, weight=w)

    # these two dictionaries have the required scores
    fairness, goodness = compute_fairness_goodness(G, 1)

    squared_error = 0.
    error = 0.
    n = 0.

    for u, v, w in test_edges:
        if u in fairness and v in goodness:
            predicted_w = fairness[u] * goodness[v]
            squared_error += (w - predicted_w)**2
            error += abs(w - predicted_w)
            n += 1

    if n==0: #Every edge in test_edges has only 1 edge connected to the graph and it was removed
        return None

    rms_error = []
    rms_error.append(math.sqrt(squared_error / n))

    for prediction_type in ("goodness", "bias", "biasc", "exclude", "excludec"):
        squared_error = 0
        error = 0
        n = 0

        for u, v, w in test_edges:
            if u in fairness and v in goodness:
                predicted_w = predict(G, u, v, goodness, prediction_type)
                squared_error += (w - predicted_w) ** 2
                error += abs(w - predicted_w)
                n += 1

        rms_error.append(math.sqrt(squared_error / n))

    return rms_error

def predict(G, u, v, goodness, prediction_type, k=5, c=3):
    """
    prediction_type is one of "bias", "goodness", or "exclude".
    """
    out_edges = G.out_edges(u, data="weight")
    if (len(out_edges) == 0
        or prediction_type == "goodness"
        or len(out_edges) <= k and (prediction_type == "exclude" or prediction_type == "excludec")):
        return goodness[v]

    if prediction_type == "biasc" or prediction_type == "excludec":
        average_error = calc_bias_csmoothing(out_edges, goodness, c)
    else:
        average_error = calc_absolute_bias(out_edges, goodness)

    prediction = goodness[v] + average_error
    if prediction < -1:
        return -1
    if prediction > 1:
        return 1

    return prediction

"""
The node exclusion algorithm returns goodness for nodes with less than k out edges.
This returns an estimation of the optimal value k.
"""
def learn_exclude_number(known_edges):
    k = 7 #Arbitrary starting point
    i = 0 #current iteration number
    maxiter = 5
    while i < maxiter:
        random.shuffle(known_edges)
        validation_edges = known_edges[:10]
        train_edges = known_edges[10:]

        G = nx.DiGraph()
        for u, v, w in train_edges:
            G.add_edge(u, v, weight=w)

        fairness, goodness = compute_fairness_goodness(G,1)

        #Generate RMSE for exclude with k-1,k,k+1 edges
        rms_error = [] #After loop will contain the RMSE of k-1,k,k+1
        for potential_k in range(k-1,k+2):
            #Try out k-1,k,and k+1

            squared_error = 0.
            error = 0.
            n = 0.
            for u, v, w in validation_edges:
                if u in fairness and v in goodness:
                    out_edges = G.out_edges(u, data="weight")
                    predicted_w = 0
                    if (len(out_edges) <= potential_k):
                        predicted_w = goodness[v]
                    else:
                        average_error = calc_absolute_bias(out_edges, goodness)

                        predicted_w = goodness[v] + average_error
                        if predicted_w < -1:
                            predicted_w = -1
                        elif predicted_w > 1:
                            predicted_w =  1

                    squared_error += (w - predicted_w) ** 2
                    error += abs(w - predicted_w)
                    n += 1

            rms_error.append(math.sqrt(squared_error / n))

        #Pick the best result
        k_adjustment = rms_error.index(min(rms_error)) - 1
        print k_adjustment
        if k_adjustment==0:
            #current value of k is best
            print i
            return k
        else:
            k+=k_adjustment

        i+=1
    print i
    return k

def learn_exclude_number2(known_edges):
    k=5
    rng=5

    random.shuffle(known_edges)
    validation_edges = known_edges[:10]
    train_edges = known_edges[10:]

    G = nx.DiGraph()
    for u, v, w in train_edges:
        G.add_edge(u, v, weight=w)

    fairness, goodness = compute_fairness_goodness(G,1)

    rms_error = [] #After loop will contain the RMSE of k-1,k,k+1
    for potential_k in range(k-rng,k+rng+1):
        #Try out k-1,k,and k+1

        squared_error = 0.
        error = 0.
        n = 0.
        for u, v, w in validation_edges:
            if u in fairness and v in goodness:
                out_edges = G.out_edges(u, data="weight")
                predicted_w = 0
                if (len(out_edges) <= potential_k):
                    predicted_w = goodness[v]
                else:
                    #######This bias term calculation should be factored out
                    average_error = calc_absolute_bias(out_edges, goodness)

                    predicted_w = goodness[v] + average_error
                    if predicted_w < -1:
                        predicted_w = -1
                    elif predicted_w > 1:
                        predicted_w =  1

                squared_error += (w - predicted_w) ** 2
                error += abs(w - predicted_w)
                n += 1

        rms_error.append(math.sqrt(squared_error / n))

    k = rms_error.index(min(rms_error)) + k - rng
    return k

def calc_absolute_bias(out_edges, goodness):
    average_error = 0.0
    for _, t, weight in out_edges:
        average_error += weight - goodness[t]

    average_error /= len(out_edges)

    return average_error

def calc_bias_csmoothing(out_edges, goodness, c):
    average_error = 0.0
    for _, t, weight in out_edges:
        average_error += weight - goodness[t]

    average_error /= (len(out_edges)+c)
    return average_error

def calc_extreme_bias(out_edges, goodness):
    average_error = 0.0
    for _, t, weight in out_edges:
        if weight/goodness[t] > 0:
            average_error += (abs(weight) - abs(goodness[t]))
        else:
            average_error += (abs(weight-goodness[t]))

    average_error /= len(out_edges)

    return average_error

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

    # print "Total iterations: %d" % iter

    return fairness, goodness

# Plot RMS Errors for 1. f*g 2. g 3. g+b 4. g+b with Exclusion
def plotRMSErrors(rms_errors,file_name):
    # Do graphing
    omit_values = [x * 10 for x in range(1, 10)]
    f_times_g_rms = []
    goodness_rms = []
    goodness_and_bias = []
    goodness_and_bias_c = []
    exclude = []
    excludec = []
    for rms_error in rms_errors:
        f_times_g_rms.append(rms_error[0])
        goodness_rms.append(rms_error[1])
        goodness_and_bias.append(rms_error[2])
        goodness_and_bias_c.append(rms_error[3])
        exclude.append(rms_error[4])
        excludec.append(rms_error[5])
    print(omit_values)
    print(f_times_g_rms)
    # plt.plot(omit_values, f_times_g_rms,'b.-', 'hello', omit_values, goodness_rms, 'k.-', 'hello', omit_values,
    #          goodness_and_bias, 'y.-', 'hello', omit_values, exclude, 'g.-', 'hello',)
    line_fg, = plt.plot(omit_values, f_times_g_rms, 'b.-', label = 'Fairness*Goodness')
    line_g, = plt.plot(omit_values, goodness_rms, 'k.-', label='Goodness')
    line_g_bias, = plt.plot(omit_values, goodness_and_bias, 'g.-', label='Goodness+Bias')
    line_g_bias_c, = plt.plot(omit_values, goodness_and_bias_c, 'r.-', label='Goodness+Bias-c')
    line_g_exclude, = plt.plot(omit_values, exclude, 'y.-', label='Exclude')
    line_g_exclude_c, = plt.plot(omit_values, excludec, 'm.-', label='Exclude-c')
    plt.legend(handles=[line_fg, line_g, line_g_bias, line_g_bias_c, line_g_exclude, line_g_exclude_c])
    plt.xlabel('Percentage of edges removed')
    plt.ylabel('Root Mean Square Error')
    plt.title(file_name)
    plt.show()
    plt.savefig('omit_results.png')

    print rms_errors


if __name__ == '__main__':
    main()

