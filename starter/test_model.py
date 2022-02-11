'''
Testing the functions in model.py file

Author: Ferruh Unlu
Date: 12/12/2021

Test 1 : Testing to see if pickle file can predict and returns rows

'''


import pandas as pd
import numpy as np
import scipy.stats
import pickle
import os
from sklearn.model_selection import train_test_split
from starter.ml.data import process_data

import starter.ml.model as md
import logging
import starter.ml.model as md

logging.basicConfig(
    filename='test_model.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

'''
Loading the same data used in training the model so that the same data can be used in testing
'''
def load_data():
    path_to_data = os.path.join(os.getcwd(), "data", "census.csv")
    data = pd.read_csv(path_to_data)
    train, test = train_test_split(data, test_size=0.20)

    return data, train, test


def get_train_data():
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    return X_train, y_train, encoder, lb

def get_test_data():
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    return X_test, y_test

def test_train_model(
        train_models,
        X_train,
        y_train,
        n_estimators=100):
    '''
    test train_models
    '''
    try:
        model = train_models(X_train, y_train, n_estimators)
        logging.info("train models successfully ran: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Testing train_models. Issue occurred while testing the model")
        raise err

    logging.info(
        "train_models function test ended. Please review the log for details.")

    return model


def test_inference(model, X):

    test_pred = md.inference(model, X)

    try:
        assert len(test_pred) > 0
        logging.info("Model returned predictions as expected. No error detected.")
    except AssertionError as err:
        logging.error(f"Error is: {0}".format(err))

    return test_pred

def test_compute_model_metrics(y_test, preds):

    precision, recall, fbeta = md.compute_model_metrics(y_test, preds)
    pr = precision * 100
    re = recall * 100

    try:
        assert pr > 70
        logging.info ("Precision is in desired range. Its value is {0}".format(pr))
    except AssertionError as err:
        logging.error(f"Precision is too low. Check your model and retrain as needed: {0}".format(pr))

    try:
        assert re > 70
        logging.info ("Recall is in desired range. Its value is {0}".format(re))
    except AssertionError as err:
        logging.error(f"Recall is too low. Check your model and retrain as needed: {0}".format(re))


if __name__ == "__main__":

    logging.info("Function testing is starting...")

    data, train, test = load_data()

    X_train, y_train, encoder, lb = get_train_data()

    #Making sure that we can train the model
    model = test_train_model(
        md.train_model,
        X_train,
        y_train,
        100)

    X_test, y_test = get_test_data()

    #Testing predictions
    preds = test_inference(model, X_test)

    #Testing the metrics to see how the model did
    test_compute_model_metrics(y_test, preds)


    logging.info("Function testing ended. Please review the log for details.")
