# Generated from: Sklearn-wrapper-plus-SimpleImputer.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
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


X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()


# ## SimpleImputer
#
# ### Mean imputation


imputer = SklearnTransformerWrapper(
    transformer = SimpleImputer(strategy='mean'),
    variables = ['LotFrontage', 'MasVnrArea'],
)

imputer.fit(X_train)


# we can find the mean values within the parameters of the
# simple imputer

imputer.transformer_.statistics_


# remove NA

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()


# ### Frequent category imputation


cols = [c for c in train_df.columns if train_df[c].dtypes=='O' and train_df[c].isnull().sum()>0]
train_df[cols].head()


imputer = SklearnTransformerWrapper(
    transformer=SimpleImputer(strategy='most_frequent'),
    variables=cols,
)

# find the most frequent category
imputer.fit(X_train)


# we can find the most frequent values within the parameters of the
# simple imputer

imputer.transformer_.statistics_


# remove NA

X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

X_train[cols].isnull().mean()


X_test[cols].head()

