import numpy as np

def r2(y, y_hat):
    res = np.sum((y - y_hat) ** 2)
    tot = np.sum((y -  np.mean(y)) ** 2)
    return 1 - (res/tot)

def closed_form(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)

def predict(X, weights, fn):
    return fn(X, weights)
