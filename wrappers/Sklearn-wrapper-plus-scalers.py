# Generated from: Sklearn-wrapper-plus-scalers.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from feature_engine.wrappers import SklearnTransformerWrapper


# data = pd.read_csv('houseprice.csv')
# data.head()

# # let's separate into training and testing set

# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

# X_train.shape, X_test.shape


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


# ## Scaling


cols = [var for var in X_train.columns if X_train[var].dtypes !='O']

cols


# let's apply the standard scaler on the above variables

scaler = SklearnTransformerWrapper(transformer = StandardScaler(),
                                    variables = cols)

scaler.fit(X_train.fillna(0))


X_train = scaler.transform(X_train.fillna(0))
X_test = scaler.transform(X_test.fillna(0))


# mean values, learnt by the StandardScaler
scaler.transformer_.mean_


# std values, learnt by the StandardScaler
scaler.transformer_.scale_


# the mean of the scaled variables is 0
X_train[cols].mean()


# the std of the scaled variables is ~1

X_train[cols].std()

