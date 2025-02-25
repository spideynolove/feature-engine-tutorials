# Generated from: LogTransformer.ipynb

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


# Impute missing values

arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)

arbitrary_imputer.fit(X_train)

# impute variables
train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)


numeric_columns = train_t.select_dtypes(include=['int64', 'float64']).columns
# numeric_columns_t = test_t.select_dtypes(include=['int64', 'float64']).columns


train_numeric = train_t[numeric_columns].copy()
# test_numeric = test_t[numeric_columns_t].copy()


train_numeric


# Define columns where zero is meaningful (count-based features)
meaningful_zeros = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 
                   'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars',
                   'PoolArea']

# Define area-based columns that need shifting
area_columns = ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 
                'TotalBsmtSF', '2ndFlrSF', 'LowQualFinSF', 'GarageArea',
                'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                'ScreenPorch', 'MiscVal']

# Create a copy of the training data
train_shifted = train_numeric.copy()

# Add small constant (1) only to area-based columns
for col in area_columns:
    train_shifted[col] = train_numeric[col] + 1

# Exclude meaningful zeros from log transformation
variables_to_transform = [col for col in train_numeric.columns if col not in meaningful_zeros]


# Now apply log transformation only to specified variables
lt = LogTransformer(base='10', variables=variables_to_transform)
lt.fit(train_shifted)


# variables that will be transformed

lt.variables_


# before transformation
train_t['GrLivArea'].hist(bins=50)
plt.title('GrLivArea')


# Before transformation
train_t['LotArea'].hist(bins=50)
plt.title('LotArea')


train_t.columns


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

