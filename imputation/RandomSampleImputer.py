# Generated from: RandomSampleImputer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Missing value imputation: RandomSampleImputer
#
#
# The RandomSampleImputer extracts a random sample of observations where data is available, and uses it to replace the NA. It is suitable for numerical and categorical variables.
#
# To control the random sample extraction, there are various ways to set a seed and ensure or maximize reproducibility.
#
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

from feature_engine.imputation import RandomSampleImputer


# Download the data from Kaggle and store it
# in the same folder as this notebook.

data = pd.read_csv('houseprice.csv')

data.head()


# Separate the data into train and test sets.

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(['Id', 'SalePrice'], axis=1),
    data['SalePrice'],
    test_size=0.3,
    random_state=0,
)

X_train.shape, X_test.shape


# ## Imputation in batch
#
# We can set the imputer to impute several observations in batch with a unique seed. This is the equivalent of setting the `random_state` to an integer in `pandas.sample()`.


# Start the imputer

imputer = RandomSampleImputer(

    # the variables to impute
    variables=['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea'],

    # the random state for reproducibility
    random_state=10,

    # equialent to setting random_state in
    # pandas.sample()
    seed='general',
)


# Stores a copy of the train set variables

imputer.fit(X_train)


# the imputer saves a copy of the variables 
# from the training set to impute new data.

imputer.X_.head()


# Check missing data in train set

X_train[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].isnull().mean()


# impute data

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# Check missing data after the transformation

train_t[['Alley', 'MasVnrType', 'LotFrontage', 'MasVnrArea']].isnull().mean()


# when using the random sample imputer, 
# the distribution of the variable does not change.

# This imputation method is useful for models that 
# are sensitive to changes in the variable distributions.

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# ## Specific seeds for each observation
#
# Sometimes, we want to guarantee that the same observation is imputed with the same value, run after run. 
#
# To achieve this, we need to always use the same seed for every particular observation. 
#
# To do this, we can use the values in neighboring variables as seed.
#
# In this case, the seed will be calculated observation per observation, either by adding or multiplying the seeding variable values, and passed to the random_state of pandas.sample(), which is used under the hood by the imputer.
# Then, a value will be extracted from the train set using that seed and  used to replace the NAN in particular observation.
#
# **To know more about how the observation per seed is used check this [notebook](https://github.com/solegalli/feature-engineering-for-machine-learning/blob/master/Section-04-Missing-Data-Imputation/04.07-Random-Sample-Imputation.ipynb)** 


imputer = RandomSampleImputer(

    # the values of these variables will be used as seed
    random_state=['MSSubClass', 'YrSold'],

    # 1 seed per observation
    seed='observation',

    # how to combine the values of the seeding variables
    seeding_method='add',
    
    # impute all variables, numerical and categorical
    variables=None,
)


# Stores a copy of the train set.

imputer.fit(X_train)


# takes a copy of the entire train set

imputer.X_


# imputes all variables.

# this procedure takes a while because it is 
# done observation per observation.

train_t = imputer.transform(X_train)
test_t = imputer.transform(X_test)


# No missing data in any variable
# after the imputation.

test_t.isnull().sum()


# when using the random sample imputer, 
# the distribution of the variable does not change

fig = plt.figure()
ax = fig.add_subplot(111)
X_train['LotFrontage'].plot(kind='kde', ax=ax)
train_t['LotFrontage'].plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')

