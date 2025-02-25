# Generated from: Sklearn-wrapper-plus-feature-selection.ipynb

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

# Read the separate files
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')

# Separate features and target in training data
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']

# For test data, you might not have the target variable
X_test = test_df.drop(['Id'], axis=1)  # Note: test data might not have SalePrice column

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


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

