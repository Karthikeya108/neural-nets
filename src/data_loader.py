import numpy as np
import pandas as pd


def load_csv_data(train_idx=50000):
    data = pd.read_csv('../data/data.csv', sep=',', header=None)
    np_data = data.values
    raw_train_data = from_csv_cols(np_data[:train_idx, ])
    raw_test_data = from_csv_cols(np_data[train_idx:, ])
    train_data = shape_data(raw_train_data)
    test_data = shape_data(raw_test_data, encode=False)
    return (train_data, test_data)


def from_csv_cols(data, num_features=784):
    features = data[:, :num_features]
    labels = data[:, num_features].flatten()
    return (features, labels)


def encode_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def shape_data(data, encode=True):
    features = [np.reshape(x, (784, 1)) for x in data[0]]
    if encode:
        labels = [encode_label(y) for y in data[1]]
    else:
        labels = data[1]
    return list(zip(features, labels))
