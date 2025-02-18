# Generated from: GeometricWidthDiscretiser.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import GeometricWidthDiscretiser


data = pd.read_csv('../data/housing.csv')   # ~ rename from train.csv
# data.head()

# # let's separate into training and testing set
# X = data.drop(["Id", "SalePrice"], axis=1)
# y = data.SalePrice

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# print("X_train :", X_train.shape)   # (1022, 79)
# print("X_test :", X_test.shape)     # (438, 79)


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


# set up the discretisation transformer
disc = GeometricWidthDiscretiser(bins=10, variables=['LotArea', 'GrLivArea'])

# fit the transformer
disc.fit(X_train)


# transform the data
train_t= disc.transform(X_train)
test_t= disc.transform(X_test)


disc.binner_dict_


fig, ax = plt.subplots(1, 2)
X_train['LotArea'].hist(ax=ax[0], bins=10);
train_t['LotArea'].hist(ax=ax[1], bins=10);

