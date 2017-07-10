import numpy as np
import math

def r2(y, y_hat):
    res = np.sum((y - y_hat) ** 2)
    tot = np.sum((y -  np.mean(y)) ** 2)
    return 1 - (res/tot)

def closed_form(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

def batch(matrix, size):
    rows = matrix.shape[0]
    for i in range(math.ceil(rows / size)):
        begin = (i * size)
        end = begin + size
        yield matrix[begin:end]

def shuffle_batch(matrix, size):
    np.random.shuffle(matrix)
    return batch(matrix, size)
