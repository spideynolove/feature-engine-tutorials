import dcor
import pandas as pd
import warnings
from sklearn.datasets import make_classification
from feature_engine.selection import DropCorrelatedFeatures
warnings.filterwarnings('ignore')
X, _ = make_classification(n_samples=1000, n_features=12, n_redundant=6,
    n_clusters_per_class=1, weights=[0.5], class_sep=2, random_state=1)
colnames = [('var_' + str(i)) for i in range(12)]
X = pd.DataFrame(X, columns=colnames)
X
dcor_tr = DropCorrelatedFeatures(variables=None, method=dcor.
    distance_correlation, threshold=0.8)
X_dcor = dcor_tr.fit_transform(X)
X_dcor
from sklearn.feature_selection import mutual_info_regression


def custom_mi(x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return mutual_info_regression(x, y)[0]


mi_tr = DropCorrelatedFeatures(variables=None, method=custom_mi, threshold=0.8)
X_mi = mi_tr.fit_transform(X)
X_mi
