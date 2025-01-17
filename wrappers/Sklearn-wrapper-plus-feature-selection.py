# Generated from: Sklearn-wrapper-plus-feature-selection.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import (
    f_regression,
    SelectKBest,
    SelectFromModel,
)

from sklearn.linear_model import Lasso

from feature_engine.wrappers import SklearnTransformerWrapper


# load dataset

data = pd.read_csv('houseprice.csv')
data.head()


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0,
)

X_train.shape, X_test.shape


# ## Select K Best


# variables to evaluate:

cols = [var for var in X_train.columns if X_train[var].dtypes !='O']

cols


# let's use select K best to select the best k variables

selector = SklearnTransformerWrapper(
    transformer = SelectKBest(f_regression, k=5),
    variables = cols)

selector.fit(X_train.fillna(0), y_train)


selector.transformer_.get_support(indices=True)


# selecteed features

X_train.columns[selector.transformer_.get_support(indices=True)]


# the transformer returns the selected variables from the list
# we passed to the transformer PLUS the remaining variables 
# in the dataframe that were not examined

X_train_t = selector.transform(X_train.fillna(0))
X_test_t = selector.transform(X_test.fillna(0))


X_test_t.head()


# ## SelectFromModel


# let's select the best variables according to Lasso

lasso = Lasso(alpha=10000, random_state=0)

sfm = SelectFromModel(lasso, prefit=False)

selector = SklearnTransformerWrapper(
    transformer = sfm,
    variables = cols)

selector.fit(X_train.fillna(0), y_train)


selector.transformer_.get_support(indices=True)


len(selector.transformer_.get_support(indices=True))


len(cols)


# the transformer returns the selected variables from the list
# we passed to the transformer PLUS the remaining variables 
# in the dataframe that were not examined

X_train_t = selector.transform(X_train.fillna(0))
X_test_t = selector.transform(X_test.fillna(0))


X_test_t.head()

