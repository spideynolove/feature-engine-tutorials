# Generated from: ArbitraryNumberImputer.ipynb

# # ArbitraryNumberImputer
#
#
# ArbitraryNumberImputer replaces NA by an arbitrary value. It works for numerical variables. The arbitrary value needs to be defined by the user.
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

from feature_engine.imputation  import ArbitraryNumberImputer


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


# ## Imputate variables with same number
#
# We will impute 2 numerical variables with the number 999.


# Check missing data

X_train[['LotFrontage', 'MasVnrArea']].isnull().mean()


# Let's create an instance of the imputer where we impute
# 2 variables with the same arbitraty number.

imputer = ArbitraryNumberImputer(
    arbitrary_number=-999,
    variables=['LotFrontage', 'MasVnrArea'],
)

imputer.fit(X_train)


# The number to use in the imputation
# is stored as parameter.

imputer.arbitrary_number


# The imputer will use the same value to impute
# all indicated variables.

imputer.imputer_dict_


# Impute variables

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

# Sanity check: the min value is the one used for 
#  the imputation

train_t[['LotFrontage', 'MasVnrArea']].min()


# The distribution of the variable
# changed with the transformation.

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ### Impute variables with different numbers
#
# We can also impute different variables with different values. In this case, we need to start the transformer with a dictionary of variable to value pairs.


# Impute different variables with different values

imputer = ArbitraryNumberImputer(
    imputer_dict={"LotFrontage": -678, "MasVnrArea": -789}
)

imputer.fit(X_train)


# In this case, the imputer_dict_ matches the 
# entered dictionary.

imputer.imputer_dict_


# Now we impute the missing data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

# Sanity check: check minimum values

train_t[['LotFrontage', 'MasVnrArea']].min()


# The distribution of the variable changed
# after the transformation.

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ## Automatically select all variables
#
# We can impute all numerical variables with the same value automatically with this transformer. We need to leave the  parameter `variables` to None.


# Let's create an instance of the imputer where we impute
# 2 variables with the same arbitraty number.

imputer = ArbitraryNumberImputer(
    arbitrary_number=-1,
)

imputer.fit(X_train)


# The imputer finds all numerical variables
# automatically.

imputer.variables_


# We find the imputation value in the dictionary

imputer.imputer_dict_


# now we impute the missing data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Sanity check:

# No numerical variable with NA is  left in the
# transformed data.

[v for v in train_t.columns if train_t[v].dtypes !=
    'O' and train_t[v].isnull().sum() > 1]


# New: we can get the name of the features in the final output
imputer.get_feature_names_out()

