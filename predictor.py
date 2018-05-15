import pandas as pd
import os
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error

def get_training_data():
    """Get the training data, from local csv or pandas repo."""
    if os.path.exists("training_edges.csv"):
        print("Found the training dataset")
        df = pd.read_csv("training_edges.csv", index_col=0)
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
    dt = DecisionTreeRegressor(min_samples_split=20, random_state=99)
    dt.fit(X, y)
    return dt

def predict(dt):
    test_df = get_test_data()
    features = list(test_df.columns[0:len(test_df.columns) -1])
    print(features)
    y_true = test_df["weight"]
    test_X = test_df[features]
    y_pred = dt.predict(test_X)
    rmse = mean_squared_error(y_true, y_pred)
    print(rmse)





if __name__ == '__main__':
    decision_tree = fit_decision_tree()
    predict(decision_tree)