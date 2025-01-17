# Generated from: Sklearn-wrapper-plus-KBinsDiscretizer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

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


cols = [var for var in X_train.columns if X_train[var].dtypes !='O']

cols


X_train[cols].hist(bins=50, figsize=(15,15))
plt.show()


# ## KBinsDiscretizer
#
# ### Equal-frequency discretization


variables = ['GrLivArea','GarageArea']

X_train[variables].isnull().mean()


# at the moment it only works if the encoding in kbinsdiscretizer
# is set to 'ordinal'

discretizer = SklearnTransformerWrapper(
    transformer = KBinsDiscretizer(
        n_bins=5, strategy='quantile', encode='ordinal'),
    variables = variables,
)

discretizer.fit(X_train)


discretizer.variables_


discretizer.transformer_


# we can find the mean values within the parameters of the
# simple imputer

discretizer.transformer_.bin_edges_


# remove NA

X_train = discretizer.transform(X_train)
X_test = discretizer.transform(X_test)


X_test['GrLivArea'].value_counts(normalize=True)


X_test['GarageArea'].value_counts(normalize=True)


X_test[variables].hist()
plt.show()

