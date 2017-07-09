import statsmodels.api as sm
prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data

X = prestige[['income', 'education']].astype(float).as_matrix()
# X = sm.add_constant(X)
y = prestige['prestige'].as_matrix()

### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###

from util import *

weights = closed_form(X, y)
y_hat = (weights[0] * X[:,0]) + (weights[1] * X[:,1])
resid = r2(y, y_hat)
