# Generated from: Sklearn-wrapper-plus-SimpleImputer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from feature_engine.wrappers import SklearnTransformerWrapper


# load house prices data set from Kaggle

data = pd.read_csv('houseprice.csv')
data.head()


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape


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


cols = [c for c in data.columns if data[c].dtypes=='O' and data[c].isnull().sum()>0]
data[cols].head()


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

