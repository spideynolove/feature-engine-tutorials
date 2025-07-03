import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from feature_engine.selection import SelectBySingleFeaturePerformance
data.head()
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=[
    'target'], axis=1), data['target'], test_size=0.3, random_state=0)
X_train.shape, X_test.shape
rf = RandomForestClassifier(n_estimators=10, random_state=1, n_jobs=4)
sel = SelectBySingleFeaturePerformance(variables=None, estimator=rf,
    scoring='roc_auc', cv=3, threshold=0.5)
sel.fit(X_train, y_train)
sel.feature_performance_
pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(
    figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('roc-auc')
len(sel.features_to_drop_)
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
X_train.shape, X_test.shape
data = pd.read_csv('../houseprice.csv')
data.shape
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
data.shape
data.head()
data.fillna(0, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['Id',
    'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)
X_train.shape, X_test.shape
rf = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=1,
    n_jobs=4)
sel = SelectBySingleFeaturePerformance(variables=None, estimator=rf,
    scoring='r2', cv=3, threshold=0.5)
sel.fit(X_train, y_train)
sel.feature_performance_
pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(
    figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('r2')
np.abs(pd.Series(sel.feature_performance_)).sort_values(ascending=False
    ).plot.bar(figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('r2 - absolute value')
len(sel.features_to_drop_)
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
X_train.shape, X_test.shape
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=['Id',
    'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)
X_train.shape, X_test.shape
rf = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=1,
    n_jobs=4)
sel = SelectBySingleFeaturePerformance(variables=None, estimator=rf,
    scoring='neg_mean_squared_error', cv=3, threshold=None)
sel.fit(X_train, y_train)
sel.feature_performance_
pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(
    figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('Negative mean Squared Error')
sel.features_to_drop_
pd.Series(sel.feature_performance_)[sel.features_to_drop_].sort_values(
    ascending=False).plot.bar(figsize=(20, 5))
