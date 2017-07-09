import statsmodels.api as sm
prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data

X = prestige[['income', 'education']].astype(float).as_matrix()
# X = sm.add_constant(X)
y = prestige['prestige'].as_matrix()

### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###

from util import *

model = sm.OLS(y, X).fit()
y_hat = model.predict(X)
weights = model.params
rss = model.rsquared
summary = model.summary()
