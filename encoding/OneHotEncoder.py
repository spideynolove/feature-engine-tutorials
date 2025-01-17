# Generated from: OneHotEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # OneHotEncoder
# Performs One Hot Encoding.
#
# The encoder can select how many different labels per variable to encode into binaries. When top_categories is set to None, all the categories will be transformed in binary variables. 
#
# However, when top_categories is set to an integer, for example 10, then only the 10 most popular categories will be transformed into binary, and the rest will be discarded.
#
# The encoder has also the possibility to create binary variables from all categories (drop_last = False), or remove the binary for the last category (drop_last = True), for use in linear models.
#
# Finally, the encoder has the option to drop the second dummy variable for binary variables. That is, if a categorical variable has 2 unique values, for example colour = ['black', 'white'], setting the parameter drop_last_binary=True, will automatically create only 1 binary for this variable, for example colour_black.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder


# Load titanic dataset from OpenML

def load_titanic():
    data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float')
    data['fare'] = data['fare'].astype('float')
    data['embarked'].fillna('C', inplace=True)
    data.drop(labels=['boat', 'body', 'home.dest'], axis=1, inplace=True)
    return data


data = load_titanic()
data.head()


X = data.drop(['survived', 'name', 'ticket'], axis=1)
y = data.survived


# we will encode the below variables, they have no missing values
X[['cabin', 'pclass', 'embarked']].isnull().sum()


''' Make sure that the variables are type (object).
if not, cast it as object , otherwise the transformer will either send an error (if we pass it as argument) 
or not pick it up (if we leave variables=None). '''

X[['cabin', 'pclass', 'embarked']].dtypes


# let's separate into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

X_train.shape, X_test.shape


# One hot encoding consists in replacing the categorical variable by a
# combination of binary variables which take value 0 or 1, to indicate if
# a certain category is present in an observation.
#
# Each one of the binary variables are also known as dummy variables. For
# example, from the categorical variable "Gender" with categories 'female'
# and 'male', we can generate the boolean variable "female", which takes 1
# if the person is female or 0 otherwise. We can also generate the variable
# male, which takes 1 if the person is "male" and 0 otherwise.
#
# The encoder has the option to generate one dummy variable per category, or
# to create dummy variables only for the top n most popular categories, that is,
# the categories that are shown by the majority of the observations.
#
# If dummy variables are created for all the categories of a variable, you have
# the option to drop one category not to create information redundancy. That is,
# encoding into k-1 variables, where k is the number if unique categories.
#
# The encoder will encode only categorical variables (type 'object'). A list
# of variables can be passed as an argument. If no variables are passed as 
# argument, the encoder will find and encode categorical variables (object type).
#
#
# #### Note:
# New categories in the data to transform, that is, those that did not appear
# in the training set, will be ignored (no binary variable will be created for them).


# ### Create all k dummy variables, top_categories=False


'''
Parameters
----------

top_categories: int, default=None
    If None, a dummy variable will be created for each category of the variable.
    Alternatively, top_categories indicates the number of most frequent categories
    to encode. Dummy variables will be created only for those popular categories
    and the rest will be ignored. Note that this is equivalent to grouping all the
    remaining categories in one group.
    
variables : list
    The list of categorical variables that will be encoded. If None, the  
    encoder will find and select all object type variables.
    
drop_last: boolean, default=False
    Only used if top_categories = None. It indicates whether to create dummy
    variables for all the categories (k dummies), or if set to True, it will
    ignore the last variable of the list (k-1 dummies).
'''

ohe_enc = OneHotEncoder(top_categories=None,
                        variables=['pclass', 'cabin', 'embarked'],
                        drop_last=False)
ohe_enc.fit(X_train)


ohe_enc.encoder_dict_


train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)

test_t.head()


# ### Selecting top_categories to encode


ohe_enc = OneHotEncoder(top_categories=2,
                        variables=['pclass', 'cabin', 'embarked'],
                        drop_last=False)
ohe_enc.fit(X_train)

ohe_enc.encoder_dict_


train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)
test_t.head()


# ### Dropping the last category for linear models


ohe_enc = OneHotEncoder(top_categories=None,
                        variables=['pclass', 'cabin', 'embarked'],
                        drop_last=True)

ohe_enc.fit(X_train)

ohe_enc.encoder_dict_


train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)

test_t.head()


# ### Automatically select categorical variables
#
# This encoder selects all the categorical variables, if None is passed to the variable argument when calling the encoder.


ohe_enc = OneHotEncoder(top_categories=None,
                        drop_last=True)

ohe_enc.fit(X_train)


# the parameter variables is None
ohe_enc.variables


# but the attribute variables_ has the categorical variables 
# that will be encoded

ohe_enc.variables_


# and we can also find which variables from those
# are binary

ohe_enc.variables_binary_


train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)

test_t.head()


# ### Automatically create 1 dummy from binary variables (sex)
#
# We can encode categorical variables that have more than 2 categories into k dummies, and, at the same time, encode categorical variables that have 2 categories only in 1 dummy. The second 1 is completely redundant.
#
# We do so as follows:


ohe_enc = OneHotEncoder(top_categories=None,
                        drop_last=False,
                        drop_last_binary=True,
                        )

ohe_enc.fit(X_train)


# the encoder dictionary
ohe_enc.encoder_dict_


# and we can also find which variables from those
# are binary

ohe_enc.variables_binary_


train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)

test_t.head()

