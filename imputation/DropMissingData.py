# Generated from: DropMissingData.ipynb

# # Missing value imputation: DropMissingData
#
# Deletes rows with missing values.
#
# DropMissingData works both with numerical and categorical variables.
#
# **For this demonstration, we use the Ames House Prices dataset produced by Professor Dean De Cock:**
#
# [Dean De Cock (2011) Ames, Iowa: Alternative to the Boston Housing
# Data as an End of Semester Regression Project, Journal of Statistics Education, Vol.19, No. 3](http://jse.amstat.org/v19n3/decock.pdf)
#
# The version of the dataset used in this notebook can be obtained from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


# ## Version


# Make sure you are using this 
# Feature-engine version.

import feature_engine

feature_engine.__version__


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from feature_engine.imputation import DropMissingData


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


# ## Drop data based on specific variables.
#
# We can drop observations that show NA in any of a subset of variables.


# Drop data when there are NA in any of the indicated variables

imputer = DropMissingData(
    variables=['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea'],
    missing_only=False,
)


imputer.fit(X_train)


# variables from which observations with NA will be deleted

imputer.variables_


# Number of observations with NA before the transformation

X_train[imputer.variables].isna().sum()


# After the transformation the rows with NA values are 
# deleted form the dataframe

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Number of observations with NA after transformation

train_t[imputer.variables].isna().sum()


# Shape of dataframe before transformation

X_train.shape


# Shape of dataframe after transformation

train_t.shape


# The "return_na_data()" method, returns a dataframe that contains
# the observations with NA. 

# That is, the portion of the data that is dropped when
# we apply the transform() method.

tmp = imputer.return_na_data(X_train)

tmp.shape


# total obs - obs with NA = final dataframe shape
#  after the transformation

1022-963


# Sometimes, it is useful to retain the observation with NA in the production environment, to log which
# observations are not being scored by the model for example.


# ## Drop data when variables contain %  of NA
#
# We can drop observations if they contain less than a required percentage of values in a subset of observations.


# Drop data if an observation contains NA in 
# 2 of the 4 indicated variables (50%).

imputer = DropMissingData(
    variables=['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea'],
    missing_only=False,
    threshold=0.5,
)


imputer.fit(X_train)


# After the transformation the rows with NA values are 
# deleted form the dataframe

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Number of observations with NA after transformation

train_t[imputer.variables].isna().sum()


# We see that not all missing observations were dropped, because we required the observation to have NA in more than 1 of the variables at the time. 


# ## Automatically select all variables
#
# We can drop obserations if they show NA in any variable in the dataset.
#
# When the parameter `variables` is left to None and the parameter `missing_only` is left to True, the imputer will evaluate observations based of all variables with missing data.
#
# When the parameter `variables` is left to None and the parameter `missing_only` is switched to False, the imputer will evaluate observations based of all variables.
#
# It is good practice to use `missing_only=True` when we set `variables=None`, so that the transformer handles the imputation automatically in a meaningful way.
#
# ### Automatically find variables with NA


# Find variables with NA

imputer = DropMissingData(missing_only=True)

imputer.fit(X_train)


# variables with NA in the train set

imputer.variables_


# Number of observations with NA

X_train[imputer.variables_].isna().sum()


# After the transformation the rows with NA are deleted form the dataframe

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Number of observations with NA after the transformation

train_t[imputer.variables_].isna().sum()


# in this case, all observations will be dropped
# because all of them show NA at least in 1 variable

train_t.shape


# ## Drop rows with % of missing data
#
# Not to end up with an empty dataframe, let's drop rows that have less than 75% of the variables with values.


# Find variables with NA

imputer = DropMissingData(
    missing_only=True,
    threshold=0.75,
)

imputer.fit(X_train)


# After the transformation the rows with NA are deleted form the dataframe

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


train_t.shape


# Now, we do have some data left.

