import random

import numpy as np


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


def inspect_data(df):
    for col in df.columns:
        unique_vals, counts = np.unique(df[col].values, return_counts=True)
        print(col, len(unique_vals), "min:", min(unique_vals), "max:", max(unique_vals))
        if len(unique_vals) <= 10:
            print(unique_vals)
            print(counts)
        print("------------------")

    print("Datframe size:", len(df))
