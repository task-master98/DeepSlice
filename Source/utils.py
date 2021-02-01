import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support
import random
from itertools import *


def compute_shape(X: np.ndarray):
    n_rows, n_cols = X.shape
    return n_rows, n_cols


def remove_random_features(X: np.ndarray, y: np.ndarray, n_features: int, model):
    rows, cols = compute_shape(X)
    feature_idx = list(range(cols))
    feature_combinations = list(combinations(feature_idx, n_features))
    X_copy = np.copy(X)
    accuracy_list = []
    features_removed = []
    f_score_list = []
    for combination in feature_combinations:
        features_removed.append(combination)
        for idx_to_remove in combination:
            X[:, idx_to_remove] = 0

        accuracy = model.evaluate(X, y)
        # f_score_list.append(calculate_metrics(X, y_test=y, model=model))
        accuracy_list.append(accuracy)
        X = X_copy
    return accuracy_list, features_removed

def feature_removed_in_itr(feature_removed: list):
    itr_dict = {
        'Iteration': np.arange(0, len(feature_removed)),
        'Feature_removed': feature_removed,
    }
    df_itr_dict = pd.DataFrame(itr_dict)
    return df_itr_dict

def plot_accuracy(accuracy_list: list, features_removed: list):
    plt.figure(figsize=(10, 10))
    plt.plot(accuracy_list)
    plt.show()
    print(features_removed)

def calculate_metrics(X_test, y_test, model):
    y_predictions = model.predict(X_test)
    y_preds = []
    for prediction in y_predictions:
        max_index = np.argmax(prediction)
        y_class = np.zeros(shape=(3,))
        y_class[max_index] = 1
        y_preds.append(y_class)
    y_preds = np.array(y_preds)
    y_preds = one_hot_to_class(y_preds)
    y_truth = one_hot_to_class(y_test)
    f_score = f1_score(y_pred=y_preds, y_true=y_truth)
    return f_score


def one_hot_to_class(y: np.ndarray):
    y_class = np.array([np.where(i == 1.0)[0][0] for i in y], dtype=int)
    return y_class

def convert_to_one_hot(y_predictions: np.ndarray):
    y_encoded = np.argmax(y_predictions, axis=1)
    return y_encoded




