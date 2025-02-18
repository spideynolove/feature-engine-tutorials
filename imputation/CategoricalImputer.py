# Generated from: CategoricalImputer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Missing value imputation: CategoricalImputer
#
#
# CategoricalImputer performs imputation of categorical variables. It replaces missing values by an arbitrary label "Missing" (default) or any other label entered by the user. Alternatively, it imputes missing data with the most frequent category.
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

from  feature_engine.imputation import CategoricalImputer


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


# These are categorical variables with missing data

X_train[['Alley', 'MasVnrType']].isnull().mean()


# Number of observations per category

X_train['MasVnrType'].value_counts().plot.bar()
plt.ylabel('Number of observations')
plt.title('MasVnrType')


# ## Imputat with string missing
#
# We replace missing data with the string "Missing".


imputer = CategoricalImputer(
    imputation_method='missing',
    variables=['Alley', 'MasVnrType'])

imputer.fit(X_train)


# We impute all variables with the
# string 'Missing'

imputer.imputer_dict_


# Perform imputation.

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Observe the new category 'Missing'

test_t['MasVnrType'].value_counts().plot.bar()

plt.ylabel('Number of observations')
plt.title('Imputed MasVnrType')


test_t['Alley'].value_counts().plot.bar()

plt.ylabel('Number of observations')
plt.title('Imputed Alley')


# ## Impute with another string
#
# We can also enter a specific string for the imputation instead of the default 'Missing'.


imputer = CategoricalImputer(
    variables='MasVnrType',
    fill_value="this_is_missing",
)


# We can also fit and transform the train set
# in one line of code
train_t = imputer.fit_transform(X_train)


# and then transform the test set
test_t = imputer.transform(X_test)


# let's check the current imputation
# dictionary

imputer.imputer_dict_


# After the imputation we see the new category

test_t['MasVnrType'].value_counts().plot.bar()

plt.ylabel('Number of observations')
plt.title('Imputed MasVnrType')


# ## Frequent Category Imputation
#
# We can also replace missing values with the most frequent category.


imputer = CategoricalImputer(
    imputation_method='frequent',
    variables=['Alley', 'MasVnrType'],
)


# Find most frequent category

imputer.fit(X_train)


# In this attribute we find the most frequent category
# per variable to impute.

imputer.imputer_dict_


# Impute variables
train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Let's count the number of observations per category
# in the original variable.

X_train['MasVnrType'].value_counts()


# note that we have a few more observations in the 
# most frequent category, which for this variable
# is 'None', after the transformation.

train_t['MasVnrType'].value_counts()


# The number of observations for `None` in `MasVnrType` increased from 609 to 614, thanks to replacing the NA with this label.


# ## Automatically select categorical variables
#
# We can impute all catetgorical variables automatically, either with a string or with the most frequent category.
#
# To do so, we need to leave the parameter `variables` to `None`.


# Impute all categorical variables with 
# the most frequent category

imputer = CategoricalImputer(imputation_method='frequent')


# with fit, the transformer identifies the categorical variables
# in the train set, and their most frequent category.
imputer.fit(X_train)

# Here we find the imputation values for each
# categorical variable.

imputer.imputer_dict_


# With transform we replace missing data.

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Sanity check:

# No categorical variable with NA is left in the
# transformed data.

[v for v in train_t.columns if train_t[v].dtypes ==
    'O' and train_t[v].isnull().sum() > 1]


# We can also return the name of the final features in
# the transformed data
imputer.get_feature_names_out()

