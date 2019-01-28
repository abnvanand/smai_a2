import numpy as np


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


# Great explanation https://youtu.be/FAr2GmWNbT0
def true_positive(cf, true_class_index):
    return cf[true_class_index][true_class_index]


def false_positive(cf, true_class_index):
    column_sum = np.sum(cf, axis=0)[true_class_index]
    return column_sum - true_positive(cf, true_class_index)


def false_negative(cf, true_class_index):
    row_sum = np.sum(cf, axis=1)[true_class_index]
    return row_sum - true_positive(cf, true_class_index)


def true_negative(cf, true_class_index):
    matrix_sum = np.sum(cf)
    row_sum = np.sum(cf, axis=1)[true_class_index]
    column_sum = np.sum(cf, axis=0)[true_class_index]
    # using inclusion exclusion principle
    return matrix_sum - row_sum - column_sum + true_positive(cf, true_class_index)


def precision_(df, true_class):
    # precision = TP / (TP+FP)
    cf, class_labels, __ = confusion_matrix(df)
    true_class_index = np.where(class_labels == true_class)[0][0]
    TP = true_positive(cf, true_class_index)
    FP = false_positive(cf, true_class_index)
    return TP / (TP + FP)


def recall_(df, true_class):
    # recall = TP / (TP + FN)
    cf, class_labels, __ = confusion_matrix(df)
    true_class_index = np.where(class_labels == true_class)[0][0]
    TP = true_positive(cf, true_class_index)
    FN = false_negative(cf, true_class_index)
    return TP / (TP + FN)


def f1_score_(df, true_class, p=None, r=None):
    if p is None:
        p = precision_(df, true_class)
    if r is None:
        r = recall_(df, true_class)

    if p + r == 0:
        return 0

    return 2 * (p * r) / (p + r)


def accuracy(df):
    correct_predictions = df[df.classification == df.label]
    accu = len(correct_predictions) / len(df)
    return accu
