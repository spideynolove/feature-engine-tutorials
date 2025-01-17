# Generated from: LogTransformer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Variable transformers : LogTransformer
#
# The LogTransformer() applies the natural logarithm or the base 10 logarithm to
# numerical variables. The natural logarithm is logarithm in base e.
#
# The LogTransformer() only works with numerical non-negative values. If the variable
# contains a zero or a negative value the transformer will return an error.
#
# **For this demonstration, we use the Ames House Prices dataset produced by Professor Dean De Cock:**
#
# Dean De Cock (2011) Ames, Iowa: Alternative to the Boston Housing
# Data as an End of Semester Regression Project, Journal of Statistics Education, Vol.19, No. 3
#
# http://jse.amstat.org/v19n3/decock.pdf
#
# https://www.tandfonline.com/doi/abs/10.1080/10691898.2011.11889627
#
# The version of the dataset used in this notebook can be obtained from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.transformation import LogTransformer


# load data

data = pd.read_csv('houseprice.csv')
data.head()


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# plot distributions before transformation

X_train['LotArea'].hist(bins=50)


# plot distributions before transformation

X_train['GrLivArea'].hist(bins=50)


# ## Log base e


# Initialzing the tansformer with log base e

lt = LogTransformer(variables=['LotArea', 'GrLivArea'], base='e')

lt.fit(X_train)


# variables that will be transformed

lt.variables_


# apply the log transform

train_t = lt.transform(X_train)
test_t = lt.transform(X_test)


# transformed variable distribution

train_t['LotArea'].hist(bins=50)


# transformed variable distribution

train_t['GrLivArea'].hist(bins=50)


# return variables to original representation

train_orig = lt.inverse_transform(train_t)
test_orig = lt.inverse_transform(test_t)


# inverse transformed variable distribution

train_orig['LotArea'].hist(bins=50)


# inverse transformed variable distribution

train_orig['GrLivArea'].hist(bins=50)


# ## Automatically select numerical variables
#
# The transformer will transform all numerical variables if no variables are specified.


# load numerical variables only

variables = ['LotFrontage', 'LotArea',
             '1stFlrSF', 'GrLivArea',
             'TotRmsAbvGrd', 'SalePrice']

data = pd.read_csv('houseprice.csv', usecols=variables)


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# Impute missing values

arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)

arbitrary_imputer.fit(X_train)

# impute variables
train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)


# transform all numerical variables with base 10

lt = LogTransformer(base='10', variables=None)

lt.fit(train_t)


# variables that will be transformed

lt.variables_


# before transformation
train_t['GrLivArea'].hist(bins=50)
plt.title('GrLivArea')


# Before transformation
train_t['LotArea'].hist(bins=50)
plt.title('LotArea')


# transform the data

train_t = lt.transform(train_t)
test_t = lt.transform(test_t)


# transformed variable

train_t['GrLivArea'].hist(bins=50)
plt.title('GrLivArea')


# transformed variable
train_t['LotArea'].hist(bins=50)
plt.title('LotArea')


# return variables to original representation

train_orig = lt.inverse_transform(train_t)
test_orig = lt.inverse_transform(test_t)


# inverse transformed variable distribution

train_orig['LotArea'].hist(bins=50)


# inverse transformed variable distribution

train_orig['GrLivArea'].hist(bins=50)

