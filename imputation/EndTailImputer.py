# Generated from: EndTailImputer.ipynb

# # EndTailImputer
#
# The EndTailImputer() replaces missing data by a value at either tail of the distribution. It automatically determines the value to be used in the imputation using the mean plus or minus a factor of the standard deviation, or using the inter-quartile range proximity rule. Alternatively, it can use a factor of the maximum value.
#
# The EndTailImputer() is in essence, very similar to the ArbitraryNumberImputer, but it selects the value to use fr the imputation automatically, instead of having the user pre-define them.
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

from feature_engine.imputation import EndTailImputer


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


# numerical variables with missing data

X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()


# The EndTailImputer can replace NA with a value at the left or right end of the distribution.
#
# In addition, it uses 3 different methods to identify the imputation values.
#
# In the following cells, we show how to use each method.
#
# ## Gaussian, right tail
#
# Let's begin by finding the values automatically at the right tail, by using the mean and the standard deviation.


imputer = EndTailImputer(
    # uses mean and standard deviation to determine the value
    imputation_method='gaussian',
    # value at right tail of distribution
    tail='right',
    # multiply the std by 3
    fold=3,
    # the variables to impute
    variables=['LotFrontage', 'MasVnrArea'],
)


# find the imputation values
imputer.fit(X_train)


# The values for the imputation
imputer.imputer_dict_


# Note that we use different values for different variables.


# impute the data
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# check we no longer have NA
train_t['LotFrontage'].isnull().sum()


# The variable distribution changed slightly with more values accumulating towards the right tail
fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ## IQR, left tail
#
# Now, we will impute variables with values at the left tail. The values are identified using the inter-quartile range proximity rule. 
#
# The IQR rule is better suited for skewed variables.


imputer = EndTailImputer(
    
    # uses the inter-quartile range proximity rule
    imputation_method='iqr',
    
    # determines values at the left tail of the distribution
    tail='left',
    
    # multiplies the IQR by 3
    fold=3,
    
    # the variables to impute
    variables=['LotFrontage', 'MasVnrArea'],
)


# finds the imputation values

imputer.fit(X_train)


# imputation values per variable

imputer.imputer_dict_


# transform the data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Check we have no NA after the transformation

train_t[['LotFrontage', 'MasVnrArea']].isnull().sum()


# The variable distribution changed with the
# transformation, with more values
# accumulating towards the left tail.

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ## Impute with the maximum value
#
# We can find imputation values with a factor of the maximum variable value.


imputer = EndTailImputer(
    
    # imputes beyond the maximum value
    imputation_method='max',
    
    # multiplies the maximum value by 3
    fold=3,
    
    # the variables to impute
    variables=['LotFrontage', 'MasVnrArea'],
)


# find imputation values

imputer.fit(X_train)


# The imputation values.

imputer.imputer_dict_


# the maximum values of the variables,
# note how the imputer multiplied them by 3
# to determine the imputation values.

X_train[imputer.variables_].max()


# impute the data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Check we have no NA  in the imputed data

train_t[['LotFrontage', 'MasVnrArea']].isnull().sum()


# The variable distribution changed with the
# transformation, with now more values
# beyond the maximum.

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ## Automatically impute all variables
#
# As with all Feature-engine transformers, the EndTailImputer can also find and impute all numerical variables in the data.


# Start the imputer

imputer = EndTailImputer()


# Check the default parameters

# how to find the imputation value
imputer.imputation_method


# which tail to use

imputer.tail


# how far out
imputer.fold


# Find variables and imputation values

imputer.fit(X_train)


# The variables to impute

imputer.variables_


#  The imputation values

imputer.imputer_dict_


# impute the data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Sanity check:

# No numerical variable with NA is  left in the
# transformed data.

[v for v in train_t.columns if train_t[v].dtypes !=
    'O' and train_t[v].isnull().sum() > 1]

