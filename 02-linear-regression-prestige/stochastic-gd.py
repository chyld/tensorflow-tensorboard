import statsmodels.api as sm
prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data

X = prestige[['income', 'education']].astype(float).as_matrix()
# X = sm.add_constant(X)
y = prestige['prestige'].as_matrix()

### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###

from util import *

def gradient_step(row, error, weights, learning_rate):
    g0 = -2 * row[0] * error
    g1 = -2 * row[1] * error

    w0 = weights[0] - (learning_rate * g0)
    w1 = weights[1] - (learning_rate * g1)

    return np.array([w0, w1])

### ++++++++++++++++++++++++ ###

# because X will be shuffled, y has to be attached
y1 = y.reshape(45,1)
matrix = np.append(X, y1, axis=1)
weights = np.array([0, 0])

### ++++++++++++++++++++++++ ###

for i in range(101):
    np.random.shuffle(matrix)
    for j, row in enumerate(matrix):
        y_hat = (weights[0] * row[0]) + (weights[1] * row[1])
        error = row[2] - y_hat
        weights = gradient_step(row, error, weights, 0.000001)
    if i % 10 == 0:
        print('i:', i, 'weights:', weights)

### ++++++++++++++++++++++++ ###
