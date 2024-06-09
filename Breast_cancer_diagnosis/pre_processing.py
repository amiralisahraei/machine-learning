from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Data Normalization
def normalization_data(data):
    minmaxScaler = MinMaxScaler()
    normalized_data = minmaxScaler.fit_transform(data)

    return normalized_data


# OneHot encoding labels
def onehot_encoding(labels):
    labels[labels == "M"] = 1
    labels[labels == "B"] = 0

    labels = np.array(labels, dtype="int")

    return labels
