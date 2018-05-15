import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
import math
from create_data_from_graph import *

def get_training_data():
    """Get the training data, from local csv or pandas repo."""
    if os.path.exists("training_edges.csv"):
        print("Found the training dataset")
        df = pd.read_csv("training_edges.csv", index_col=0)
        print("done reading csv")
    else:
        print("COULDNT FIND TRAINING DATASET")
    return df

def get_test_data():
    """Get the training data, from local csv or pandas repo."""
    if os.path.exists("test_edges.csv"):
        print("Found the testing dataset")
        df = pd.read_csv("test_edges.csv", index_col=0)
    else:
        print("COULDNT FIND TESTING DATASET")
    return df


def fit_decision_tree():
    train_df = get_training_data()
    features = list(train_df.columns[0:len(train_df.columns) - 1])
    y = train_df["weight"]
    X = train_df[features]
    print("wtf")
    dt = DecisionTreeRegressor(min_samples_split=60, random_state=99)
    dt.fit(X, y)
    print("done training")
    return dt

def predict_decision_tree(dt):
    test_df = get_test_data()
    features = list(test_df.columns[0:len(test_df.columns) -1])
    print(features)
    y_true = test_df["weight"]
    test_X = test_df[features]
    y_pred = dt.predict(test_X)
    mean_error = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    # print(rmse)
    # print(mean_error)

def predict_kernelized_ridge_regression():
    # Train kernel
    train_df = get_training_data()
    features = list(train_df.columns[0:len(train_df.columns) - 1])
    y = train_df["weight"]
    X = train_df[features]
    clf = KernelRidge(alpha=1.0)
    print("wtffff")
    clf.fit(X, y)

    # Test
    test_df = get_test_data()
    y_true = test_df["weight"]
    test_X = test_df[features]
    y_pred = clf.predict(test_X)
    mean_error = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(rmse, mean_error)
    return rmse, mean_error



if __name__ == '__main__':
    file_name = sys.argv[1]
    percentage_omit = 0.1
    rmses = []
    abs_error = []
    for x in range(0, 8):
        print(percentage_omit)
        run(file_name, percentage_omit)
        rmse, mean_error = predict_kernelized_ridge_regression()
        rmses.append(rmse)
        abs_error.append(mean_error)

    print(rmses)
    print(abs)


