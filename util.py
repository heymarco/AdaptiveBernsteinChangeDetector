import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess(x: np.ndarray):
    return MinMaxScaler().fit_transform(x)
