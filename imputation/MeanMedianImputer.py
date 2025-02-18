# Generated from: MeanMedianImputer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Missing value imputation: MeanMedianImputer
#
# The MeanMedianImputer() replaces missing data by the mean or median value of the variable. 
#
# It works only with numerical variables.
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

from  feature_engine.imputation import MeanMedianImputer


# ## Load data


# # Download the data from Kaggle and store it in the same folder as this notebook.
# data = pd.read_csv('../data/housing.csv')
# data.head()

# # Separate the data into train and test sets.
# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop(['Id', 'SalePrice'], axis=1),
#     data['SalePrice'],
#     test_size=0.3,
#     random_state=0,
# )

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


# ## Check missing data


# Numerical variables with missing data

X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()


# ## Imputation with the median
#
# Let's start by imputing missing data in 2 variables with their median.


# Set up the imputer.

imputer = MeanMedianImputer(
    imputation_method='median',
    variables=['LotFrontage', 'MasVnrArea'],
)


# Find median values

imputer.fit(X_train)


# Dictionary with the imputation values for each variable.

imputer.imputer_dict_


# Let's corroborate that the dictionary 
# contains the median values of the variables.

X_train[['LotFrontage', 'MasVnrArea']].median()


# impute the data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Check we no longer have NA

train_t[['LotFrontage', 'MasVnrArea']].isnull().sum()


# The variable distribution changed slightly with
# more values accumulating towards the median 
# after the imputation.

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ## Automatically select all numerical variables
#
# Let's now impute all numerical variables with the mean.
#
# If we leave the parameter `variables` to `None`, the transformer identifies and imputes all numerical variables.


# Set up the imputer

imputer = MeanMedianImputer(
    imputation_method='mean',
)


# Find numerical variables and their mean.

imputer.fit(X_train)


# Numerical variables identified.

imputer.variables_


# The imputation value, the mean, for each variable

imputer.imputer_dict_


# impute the data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

# the numerical variables do not have NA after
# the imputation.

test_t[imputer.variables_].isnull().sum()

