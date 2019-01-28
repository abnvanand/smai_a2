# import random
# def train_test_split(df, test_size, random_state=None):
#     if random_state is not None:
#         random.seed(random_state)

#     test_size = round(len(df) * test_size)
#     test_indexes = random.sample(population=range(len(df)), k=test_size)
#     test_df = df.loc[test_indexes]
#     train_df = df.drop(test_indexes)
#     return train_df, test_df

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def train_test_split(df, test_size, random_state=None):
    """Splits data into training and testing sets.
    test_size must be fractional eg 0.2 for 20% split"""

    if random_state is not None:
        # Seed to generate same pseudo-random numbers everytime to make it reproducible.
        random.seed(random_state)

    test_size = round(test_size * len(df))  # change proportion to actual number of rows

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices, :]
    train_df = df.drop(test_indices)

    return train_df, test_df


def precision(df):
    # precision = TP / (TP+FP)
    # TP = number_of_people_correctly_identified_as_leaving_the_company
    # FP = number_of_people_INcorrectly_identified_as_leaving_the_company but are actually not leaving
    true_positive = len(df[(df.label == 1) & (df.classification == df.label)])
    false_positive = len(df[(df.classification == 1) & (df.label == 0)])

    if true_positive + false_positive == 0:
        return 0

    return true_positive / (true_positive + false_positive)


def recall(df):
    # recall = TP/(TP+FN)
    # TP = number_of_people_correctly_identified_as_leaving_the_company
    # FN = number_of_people_actually_leaving the company but identified as not leaving
    true_positive = len(df[(df.label == 1) & (df.classification == df.label)])
    false_negative = len(df[(df.label == 1) & (df.classification != df.label)])

    if true_positive + false_negative == 0:
        return 0

    return true_positive / (true_positive + false_negative)


def f1_score(df, p=None, r=None):
    if p is None:
        p = precision(df)
    if r is None:
        r = recall(df)

    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)


def accuracy(df):
    correct_predictions = df[df.classification == df.label]
    accu = len(correct_predictions) / len(df)
    return accu


def confusion_matrix(df):
    """Creates confusion matrix with true labels along rows and predicted labels along columns.

    Assumes df contains columns "label"=>True labels and "classification"=>Predicted labels"""
    rows, true_counts = np.unique(df["label"].values, return_counts=True)
    cols, predicted_counts = np.unique(df["label"].values, return_counts=True)

    matrix = np.ndarray(shape=(len(rows), len(cols)), dtype=float)
    for ri, row in enumerate(rows):
        for ci, col in enumerate(cols):
            matrix[ri][ci] = len(df[(df.label == row) & (df.classification == col)])

    return matrix, rows, cols


def plot_confusion_heatmap(df, xlabel='Predicted labels', ylabel='True labels', xticks_rotation=45, yticks_rotation=0,
                           fontsize=14):
    matrix, class_names, _ = confusion_matrix(df)

    df_cm = pd.DataFrame(
        matrix, index=class_names, columns=class_names,
    )
    heatmap = sns.heatmap(df_cm, annot=True, fmt='g')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=xticks_rotation, ha='right',
                                 fontsize=fontsize)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=yticks_rotation, ha='right',
                                 fontsize=fontsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
