import numpy as np

X = np.array([
    [1, 1250],
    [1, 1700],
    [1, 1400],
    [1, 1900]
])

y = np.array([111000, 222000, 155000, 277000])

### CLOSED FORM ###

Xt = X.transpose()
XtX = Xt.dot(X)
XtXn1 = np.linalg.inv(XtX)
Xty = Xt.dot(y)
cf = XtXn1.dot(Xty)

### ----------- ###

def predict(X, weights):
    return X.dot(weights)

def cost(y, y_hat):
    return np.sum((y - y_hat) ** 2)

def gradient_step(X, y, weights, alpha):
    g0 = 0
    g1 = 0
    N = len(y)
    y_hat = predict(X, weights)
    dy = y - y_hat
    for n in range(N):
        x0 = X[n,0]
        x1 = X[n,1]
        g0 += -2 * x0 * dy[n]
        g1 += -2 * x1 * dy[n]
        print('x0:', x0, 'x1:', x1, 'g0:', g0, 'g1:', g1, 'dy:', dy[n])
    dw0 = g0 * 0.0001
    dw1 = g1 * 0.00000001
    print('---------------------------------------------------------------------> dw0:', dw0, 'dw1:', dw1)
    w0 = weights[0] - dw0
    w1 = weights[1] - dw1
    return np.array([w0, w1])

def train():
    weights = np.array([0, 0])
    for i in range(30):
        weights = gradient_step(X, y, weights, 0.0001)
        amt = cost(y, predict(X, weights))
        print('weights:', weights, 'cost:', amt/1000000)

train()

print('perfect --------------------------')
amt = cost(y, predict(X, cf))
print('weights:', cf, 'cost:', amt/1000000)

for m in range(-300, 300, 10):
    for b in range(-300_000, 300_000, 10_000):
        y_hat = predict(X, np.array([b, m]))
        c = cost(y, y_hat) / 100000000
        if c < 25:
            print('b:', b, 'm:', m, 'cost:', c)
