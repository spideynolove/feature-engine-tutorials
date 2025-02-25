# Generated from: Select-by-Single-Feature-Performance.ipynb

# ## Univariate Single Performance
#
# - Train a ML model per every single feature
# - Determine the performance of the models
# - Select features if model performance is above a certain threshold


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error

from feature_engine.selection import SelectBySingleFeaturePerformance


# ## Classification


# data.shape


data.head()


# **Important**
#
# In all feature selection procedures, it is good practice to select the features by examining only the training set. And this is to avoid overfit.


# separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['target'], axis=1),
    data['target'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# set up a machine learning model
rf = RandomForestClassifier(
    n_estimators=10, random_state=1, n_jobs=4)

# set up the selector
sel = SelectBySingleFeaturePerformance(
    variables=None,
    estimator=rf,
    scoring="roc_auc",
    cv=3,
    threshold=0.5)

# find predictive features
sel.fit(X_train, y_train)


#  the transformer stores a dictionary of feature:metric pairs
# in this case is the roc_auc of each individual model

sel.feature_performance_


# we can plot feature importance sorted by importance

pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('roc-auc')


# the features that will be removed

len(sel.features_to_drop_)


# remove non-prective features

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape


# ## Regression
#
# ### with r2 and user specified threshold


# load dataset

data = pd.read_csv('../houseprice.csv')

data.shape


# I will use only numerical variables
# select numerical columns:

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_vars = list(data.select_dtypes(include=numerics).columns)
data = data[numerical_vars]
data.shape


data.head()


# fill missing values
data.fillna(0, inplace=True)


# separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['Id','SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# set up the machine learning model
rf = RandomForestRegressor(
    n_estimators=10, max_depth=2, random_state=1, n_jobs=4)

# set up the selector
sel = SelectBySingleFeaturePerformance(
    variables=None,
    estimator=rf,
    scoring="r2",
    cv=3,
    threshold=0.5)

# find predictive features
sel.fit(X_train, y_train)


# the transformer stores a dictionary of feature:metric pairs
# notice that the r2 can be positive or negative.
# the selector selects based on the absolute value

sel.feature_performance_


pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('r2')


# same plot but taking the absolute value of the r2

np.abs(pd.Series(sel.feature_performance_)).sort_values(ascending=False).plot.bar(figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('r2 - absolute value')


# the features that will be removed

len(sel.features_to_drop_)


# select features in the dataframes

X_train = sel.transform(X_train)
X_test = sel.transform(X_test)

X_train.shape, X_test.shape


# ### Automatically select threshold
#
# If we leave the threshold to None, the threshold will be automatically specified as the mean of performance of all features.


# separate train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['Id','SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


# set up the machine learning model
rf = RandomForestRegressor(
    n_estimators=10, max_depth=2, random_state=1, n_jobs=4)

# set up the selector
sel = SelectBySingleFeaturePerformance(
    variables=None,
    estimator=rf,
    scoring="neg_mean_squared_error",
    cv=3,
    threshold=None)

# find predictive features
sel.fit(X_train, y_train)


# the transformer stores a dictionary of feature:metric pairs
# the selector will select those features with neg mean squared error
# bigger than the mean of the neg squared error of all features

sel.feature_performance_


pd.Series(sel.feature_performance_).sort_values(ascending=False).plot.bar(figsize=(20, 5))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('Negative mean Squared Error')


# the features that will be dropped
sel.features_to_drop_


# note that these features have the biggest negative mean squared error
pd.Series(sel.feature_performance_)[sel.features_to_drop_].sort_values(ascending=False).plot.bar(figsize=(20, 5))

