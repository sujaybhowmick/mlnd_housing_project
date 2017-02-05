import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def read_data(data_file):
    data = pd.read_csv(data_file)
    return data

def get_prices_features(data):
    return (data['MEDV'], data.drop('MEDV', axis = 1))

def get_train_test_data(prices, features):
    return train_test_split(features, prices, test_size=0.2, random_state=1)

def performance_metric(y_true, y_predict):
    score = r2_score(y_true, y_predict)
    return score

def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Shuffle and Split to obtain cross validation sets
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0).get_n_splits(X, y)

    # Create decesion tree regressor
    regressor = DecisionTreeRegressor()

    # create depth parameters (1-10)
    depth_params = {"max_depth": list(range(1, 11))}

    # define a scoring function
    score_fn = make_scorer(performance_metric)

    # Grid Search Cross Validation

    grid_cv = GridSearchCV(regressor, depth_params, scoring=score_fn, cv=cv_sets)

    grid_cv.fit(X, y)

    return grid_cv.best_estimator_


def main():
    cmd_args = sys.argv
    if len(cmd_args) < 3:
        print "Usage: python {} <data_file> <client_data_file>".format(__file__)
        exit(1)

    data_file =  cmd_args[1]
    data = read_data(data_file)
    prices, features = get_prices_features(data)

    # Spilt the data in to training and testing sets
    X_train, X_test, y_train, y_test = get_train_test_data(prices, features)
    estimator = fit_model(X_train, y_train)
    client_data_file = cmd_args[2]
    client_data = read_data(client_data_file)
    #[[5, 17, 15],
    #    [4, 32, 22],
    #    [8, 3, 12]]

    predictions = estimator.predict(client_data)
    for i, price in enumerate(predictions):
        print "Predicted Selling price for client {}'s home is ${:,.2f}". format(i+1, price)

if __name__ == '__main__':
    main()
