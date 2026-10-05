# %%
from functools import partial

import linear_model as skl
import model_selection as skm
import numpy as np
import pandas as pd
import sklearn
from ISLP import load_data
from ISLP.models import ModelSpec as MS
from ISLP.models import Stepwise, sklearn_selected, sklearn_selection_path
from l0bnb import fit_path
from matplotlib.pyplot import subplots
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

# %%
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.api import OLS

# %%
Hitters = load_data("Hitters")
np.isnan(Hitters["Salary"]).sum()
# %%
Hitters.head()
# %%
Hitters = Hitters.dropna()
Hitters.shape


# %%
def nCp(sigma2, estimator, X, Y):
    "Negative Cp statistic"
    n, p = X.shape
    Yhat = estimator.predict(X)
    RSS = np.sum((Y - Yhat) ** 2)
    return -(RSS + 2 * p * sigma2) / n


# %%
design = MS(Hitters.columns.drop("Salary")).fit(Hitters)
Y = np.array(Hitters["Salary"])
X = design.transform(Hitters)
sigma2 = OLS(Y, X).fit().scale
# %%
neg_Cp = partial(nCp, sigma2)
# %%
strategy = Stepwise.first_peak(design, direction="forward", max_terms=len(design.terms))
# %%
hitters_MSE = sklearn_selected(OLS, strategy)
hitters_MSE.fit(Hitters, Y)
hitters_MSE.selected_state_

# %%
hitters_MSE = sklearn_selected(OLS, strategy, scoring=neg_Cp)
hitters_MSE.fit(Hitters, Y)
hitters_MSE.selected_state_
