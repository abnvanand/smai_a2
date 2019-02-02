import numpy as np

def get_k_nearest_neighbours(training_df, test_point, k, distance_measure_algo):
    """Returns k nearest neighbours of test_point, among training_points"""
    dimensions = len(test_point) - 1  # -1 to not consider class label when calculating distance
    dists = np.apply_along_axis(distance_measure_algo, 1, training_df, test_point, dimensions)
    distances = np.column_stack((dists, training_df))

    distances = distances[distances[:, 0].argsort()]
    k_neighbours = distances[:k, 1:]
    return k_neighbours


def best_class(neighbours):
    """Returns most prominent class label among the neighbours."""
    class_labels = neighbours[:,-1]
    unique_labels, counts = np.unique(class_labels, return_counts=True)
    index = counts.argmax()
    return unique_labels[index]


def classify_example(example, training_df, k, distance_measure_algo):
    """Predicts the class label for the example data."""
    neighbours = get_k_nearest_neighbours(training_df, example.values, k, distance_measure_algo)
    return best_class(neighbours)


def predict(test_df, train_df, k, distance_measure_algo):
    """Adds a classification column to the test dataframe. 
    This classification column contains the predictions made by the columns."""
    predictions = test_df.apply(classify_example, axis=1, args=(train_df, k,distance_measure_algo))
    predictions.name="classification"
    return predictions