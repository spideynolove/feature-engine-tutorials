# Generated from: YeoJohnsonTransformer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Variable transformers : YeoJohnsonTransformer
#
# The YeoJohnsonTransformer() applies the Yeo-Johnson transformation to the
# numerical variables.
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
from feature_engine.transformation import YeoJohnsonTransformer


# load data

data = pd.read_csv('houseprice.csv')
data.head()


# let's separate into training and testing set

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# initialize transformer to transform 2 variables

yjt = YeoJohnsonTransformer(variables = ['LotArea', 'GrLivArea'])

# find otpimal lambdas for the transformation
yjt.fit(X_train)


# these are the lambdas for the YeoJohnson transformation

yjt.lambda_dict_


# transform variables

train_t = yjt.transform(X_train)
test_t = yjt.transform(X_test)


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


# ## Automatically select numerical variables
#
# Before using YeoJohnsonTransformer we need to ensure that numerical variables do not have missing data.


# impute missing data

arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)

arbitrary_imputer.fit(X_train)

train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)


# intializing transformer to transform all variables

yjt = YeoJohnsonTransformer()

yjt.fit(train_t)


# Note, the run time error is because we are trying to transform integers.


# variables that will be transformed
# (these are the numerical variables in the dataset)

yjt.variables_


# these are the parameters for YeoJohnsonTransformer

yjt.lambda_dict_


# transform  variables
train_t = yjt.transform(train_t)
test_t = yjt.transform(test_t)

