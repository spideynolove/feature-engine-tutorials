# Generated from: Smart-Correlation-Selection.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# ## Custom methods in `SmartCorrelatedSelection`
#
# In this tutorial we show how to pass a custom method to `SmartCorrelatedSelection` using the association measure [Distance Correlation](https://m-clark.github.io/docs/CorrelationComparison.pdf) from the python package [dcor](https://dcor.readthedocs.io/en/latest/index.html). Install `dcor` before starting the tutorial
#
# ```
# !pip install dcor
# ```


import pandas as pd
import dcor
import warnings

from sklearn.datasets import make_classification
from feature_engine.selection import SmartCorrelatedSelection

warnings.filterwarnings('ignore')


X, _ = make_classification(
    n_samples=1000,
    n_features=12,
    n_redundant=6,
    n_clusters_per_class=1,
    weights=[0.50],
    class_sep=2,
    random_state=1
)

colnames = ['var_'+str(i) for i in range(12)]
X = pd.DataFrame(X, columns=colnames)


dcor_tr = SmartCorrelatedSelection(
    variables=None,
    method=dcor.distance_correlation,
    threshold=0.75,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
)

X_dcor = dcor_tr.fit_transform(X)
X_dcor


# In the next example, we use the function [sklearn.feature_selection.mutual_info_regression](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression) to calculate the Mutual Information between two numerical variables.
#
# As the callable should take as input two 1d ndarrays and output a float value, we define a custom function calling the sklearn method.


from sklearn.feature_selection import mutual_info_regression

def custom_mi(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return mutual_info_regression(x, y)[0] # should return a float value


mi_tr = SmartCorrelatedSelection(
    variables=None,
    method=custom_mi,
    threshold=0.75,
    missing_values="raise",
    selection_method="variance",
    estimator=None,
)

X_mi = mi_tr.fit_transform(X)
X_mi

