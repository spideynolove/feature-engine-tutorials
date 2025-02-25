# Generated from: AddMissingIndicator.ipynb

# # AddMissingIndicator
#
#
# AddMissingIndicator adds additional binary variables indicating missing data (thus, called missing indicators). The binary variables take the value 1 if the observation's value is missing, or 0 otherwise. AddMissingIndicator adds 1 binary variable per variable.
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
from sklearn.pipeline import Pipeline

from feature_engine.imputation import (
    AddMissingIndicator,
    MeanMedianImputer,
    CategoricalImputer,
)


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


# ## Add indicators
#
# We will add indicators to 4 variables with missing data.


# Check missing data

X_train[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].isnull().mean()


# Start the imputer with the variables for which
# we want indicators.

imputer = AddMissingIndicator(
    variables=['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea'],
)

imputer.fit(X_train)


# the variables for which missing 
# indicators will be added.

imputer.variables_


# Check the added indicators. They take the name of
# the variable underscore na

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

train_t[['Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']].head()


# Note that the original variables still have missing data.

train_t[['Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']].mean()


# ## Indicators plus imputation
#
# We normally add missing indicators and impute the original variables with the mean or median if the variable is numerical, or with the mode if the variable is categorical. So let's do that.


# Check variable types

X_train[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].dtypes


# The first 2 variables are categorical, so I will impute them with the most frequent category. The last variables are numerical, so I will impute with the median.


# Create a pipeline with the imputation strategy

pipe = Pipeline([
    ('indicators', AddMissingIndicator(
        variables=['Alley', 'MasVnrType',
                   'LotFrontage', 'MasVnrArea'],
    )),

    ('imputer_num', MeanMedianImputer(
        imputation_method='median',
        variables=['LotFrontage', 'MasVnrArea'],
    )),

    ('imputer_cat', CategoricalImputer(
        imputation_method='frequent',
        variables=['Alley', 'MasVnrType'],
    )),
])


# With fit() the transformers learn the 
# required parameters.

pipe.fit(X_train)


# We can look into the attributes of the
# different transformers.

# Check the variables that will take indicators.
pipe.named_steps['indicators'].variables_


# Check the median values for the imputation.

pipe.named_steps['imputer_num'].imputer_dict_


# Check the mode values for the imputation.

pipe.named_steps['imputer_cat'].imputer_dict_


# Now, we transform the data.

train_t = pipe.transform(X_train)
test_t = pipe.transform(X_test)


# Lets' look at the transformed variables.

# original variables plus indicators
vars_ = ['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea',
         'Alley_na', 'MasVnrType_na', 'LotFrontage_na', 'MasVnrArea_na']

train_t[vars_].head()


# After the transformation, the variables do not
# show missing data

train_t[vars_].isnull().sum()


# ## Automatically select the variables
#
# We have the option to add indicators to all variables in the dataset, or to all variables with missing data. AddMissingIndicator can select which variables to transform automatically.
#
# When the parameter `variables` is left to None and the parameter `missing_only` is left to True, the imputer add indicators to all variables with missing data.
#
# When the parameter `variables` is left to None and the parameter `missing_only` is switched to False, the imputer add indicators to all variables.
#
# It is good practice to use `missing_only=True` when we set `variables=None`, so that the transformer handles the imputation automatically in a meaningful way.
#
# ### Automatically find variables with NA


# With missing_only=True, missing indicators will only be added
# to those variables with missing data found during the fit method
# in the train set


imputer = AddMissingIndicator(
    variables=None,
    missing_only=True,
)

# finds variables with missing data
imputer.fit(X_train)


# The original variables argument was None

imputer.variables


# In variables_ we find the list of variables with NA
# in the train set

imputer.variables_


len(imputer.variables_)


# We've got 19 variables with NA in the train set.


# After transforming the dataset, we see more columns
# corresponding to the missing indicators.

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

X_train.shape, train_t.shape


# Towards the right, we find the missing indicators.

train_t.head()


# ## Add indicators to all variables


# We can, in practice, set up the indicator to add
# missing indicators to all variables

imputer = AddMissingIndicator(
    variables=None,
    missing_only=False,
)

imputer.fit(X_train)


# the attribute variables_ now shows all variables
# in the train set.

len(imputer.variables_)


# After transforming the dataset,
# we obtain double the number of columns

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)

X_train.shape, train_t.shape


# ## Automatic imputation
#
# We can automatically impute missing data in numerical and categorical variables, letting the imputers  find out which variables to impute.
#
# We need to set the parameter variables to None in all imputers. None is the default value, so we can simply omit the parameter when initialising the transformers.


# Create a pipeline with the imputation strategy

pipe = Pipeline([
    # add indicators to variables with NA
    ('indicators', AddMissingIndicator(
        missing_only=True,
    )),
    # impute all numerical variables with the median
    ('imputer_num', MeanMedianImputer(
        imputation_method='median',
    )),
    # impute all categorical variables with the mode
    ('imputer_cat', CategoricalImputer(
        imputation_method='frequent',
    )),
])


# With fit() the transformers learn the required parameters.
pipe.fit(X_train)


# We can look into the attributes of the different transformers.
# Check the variables that will take indicators.
pipe.named_steps['indicators'].variables_


# Check the median values for the imputation.
pipe.named_steps['imputer_num'].imputer_dict_


# Check the mode values for the imputation.
pipe.named_steps['imputer_cat'].imputer_dict_


# Now, we transform the data.
train_t = pipe.transform(X_train)
test_t = pipe.transform(X_test)


# We should see a complete case dataset
train_t.isnull().sum()


# Sanity check
[v for v in train_t.columns if train_t[v].isnull().sum() > 1]

