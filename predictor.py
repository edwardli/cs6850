import pandas as pd
import os
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import math
from sklearn import datasets, linear_model
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import RFE




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
    dt = DecisionTreeRegressor(min_samples_split=60, random_state=99)
    dt.fit(X, y)
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
    print(rmse)
    print(mean_error)


def fit_random_forest():
    train_df = get_training_data()
    features = list(train_df.columns[0:len(train_df.columns) - 1])
    y = train_df["weight"]
    X = train_df[features]
    dt = RandomForestRegressor(max_depth=6, random_state=0)
    dt.fit(X, y)
    return dt


def predict_random_forest(dt):
    test_df = get_test_data()
    features = list(test_df.columns[0:len(test_df.columns) -1])
    print(features)
    y_true = test_df["weight"]
    test_X = test_df[features]
    y_pred = dt.predict(test_X)
    mean_error = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(rmse)
    print(mean_error)

def predict_kernelized_ridge_regression():
    # Train kernel
    train_df = get_training_data()
    features = list(train_df.columns[0:len(train_df.columns) - 1])
    y = train_df["weight"]
    X = train_df[features]
    clf = KernelRidge(alpha=1.0)
    print("Fitting Kernelized Linear Regression")
    clf.fit(X, y)
    print("Done Fitting Kernelized Linear Regression")

    # Test
    test_df = get_test_data()
    y_true = test_df["weight"]
    test_X = test_df[features]
    y_pred = clf.predict(test_X)
    mean_error = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(rmse)
    print(mean_error)

def predict_linear_regression():
    train_df = get_training_data()
    features = list(train_df.columns[0:len(train_df.columns) - 1])
    y_train = train_df["weight"]
    X_train = train_df[features]
    X_train_scaled = preprocessing.scale(X_train)
    lm = LinearRegression()
    print("Fitting Linear Regression")
    lm.fit(X_train_scaled, y_train)
    print("Done Linear Regression")

    # Test
    test_df = get_test_data()
    y_true = test_df["weight"]
    test_X = test_df[features]
    X_test_scaled = preprocessing.scale(test_X)
    y_pred = lm.predict(X_test_scaled)
    print(y_pred)
    mean_error = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(rmse)
    print(mean_error)
    plt.hist(y_pred, bins=[x*0.1 for x in range(-10, 11)])
    plt.xlabel("Predicted Weight")
    plt.ylabel("Count")
    plt.title("Distribution of predictions on RFA")
    # plt.show()

    # Estimate the top coefficients of the linear model
    estimator = LinearRegression()
    selector = RFE(estimator, 5, step=1)
    selector = selector.fit(X_train_scaled, y_train)
    print("TOP COEFFICIENTS")
    print(selector.n_features_)
    print(selector.support_)
    print(selector.ranking_)



    # Random code from https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression to
    # get the P value
    # X2 = sm.add_constant(X_train_scaled)
    # est = sm.OLS(y_train, X2)
    # est2 = est.fit()
    # print(est2.summary())
    #
    # params = np.append(lm.intercept_, lm.coef_)
    # predictions = lm.predict(X_train_scaled)
    #
    # newX = pd.DataFrame({"Constant": np.ones(len(X_train_scaled))}).join(pd.DataFrame(X_train_scaled))
    # MSE = (sum((y_train - predictions) ** 2)) / (len(newX) - len(newX.columns))
    #
    #
    # var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
    # sd_b = np.sqrt(var_b)
    # ts_b = params / sd_b
    #
    # p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b]
    #
    # sd_b = np.round(sd_b, 3)
    # ts_b = np.round(ts_b, 3)
    # p_values = np.round(p_values, 3)
    # params = np.round(params, 4)
    #
    # myDF3 = pd.DataFrame()
    # coefficients = ['in_degree_u', 'in_degree_v', 'out_degree_u', 'out_degree_v', 'num_common_out', 'num_common_in', 'ratio_1', 'ratio_2',
    #                 'ratio_3', 'ratio_4', 'avg_ratings_into_v', 'avg_ratings_out_of_u', 'fairness_u', 'fairness_v', 'goodness_u', 'goodness_v','f_times_g']
    # myDF3["Predictor"], myDF3["Coefficients"], myDF3["Standard Errors"], myDF3["Probabilites"] = [coefficients, params, sd_b,
    #                                                                                              p_values]
    # print(myDF3)



if __name__ == '__main__':
    # decision_tree = fit_decision_tree()
    # predict_decision_tree(decision_tree)
    #predict_kernelized_ridge_regression()
    # predict_linear_regression()
    random_forest = fit_random_forest()
    predict_random_forest(random_forest)



