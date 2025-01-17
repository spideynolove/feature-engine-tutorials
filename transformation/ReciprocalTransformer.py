# Generated from: ReciprocalTransformer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Variable transformers : ReciprocalTransformer
#
# The ReciprocalTransformer() applies the reciprocal transformation 1 / x
# to numerical variables.
#
# The ReciprocalTransformer() only works with numerical variables with non-zero
# values. If a variable contains the value  the transformer will raise an error.
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
from feature_engine.transformation import ReciprocalTransformer


# load data

data = pd.read_csv('houseprice.csv')
data.head()


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# transform 2 variables

rt = ReciprocalTransformer(variables = ['LotArea', 'GrLivArea'])

rt.fit(X_train)


# variables to transform

rt.variables_


# transforming variables
train_t = rt.transform(X_train)
test_t = rt.transform(X_test)


# variable before transformation
X_train['GrLivArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('GrLivArea')


# transformed variable
train_t['GrLivArea'].hist(bins=50)
plt.title('Transformed variable')
plt.xlabel('GrLivArea')


# tvariable before transformation
X_train['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')


# transformed variable
train_t['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')


# return variables to original representation

train_orig = rt.inverse_transform(train_t)
test_orig = rt.inverse_transform(test_t)


# inverse transformed variable distribution

train_orig['LotArea'].hist(bins=50)


# inverse transformed variable distribution

train_orig['GrLivArea'].hist(bins=50)


# ## Automatically select numerical variables
#
# We cannot do reciprocal transformation when the variable values are zero so we will use only positive variables for this demo.


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


# reciprocal transformation

rt = ReciprocalTransformer()

rt.fit(train_t)


# variables to transform
rt.variables_


# before transforming 

train_t['GrLivArea'].hist(bins=50)


# before transforming 
train_t['LotArea'].hist(bins=50)


# transform variables
train_t = rt.transform(train_t)
test_t = rt.transform(test_t)


# transformed variable
train_t['GrLivArea'].hist(bins=50)


# transformed variable
train_t['LotArea'].hist(bins=50)


# return variables to original representation

train_orig = rt.inverse_transform(train_t)
test_orig = rt.inverse_transform(test_t)


# inverse transformed variable distribution

train_orig['LotArea'].hist(bins=50)


# inverse transformed variable distribution

train_orig['GrLivArea'].hist(bins=50)

