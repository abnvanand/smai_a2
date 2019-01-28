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
