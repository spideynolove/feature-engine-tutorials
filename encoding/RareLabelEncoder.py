# Generated from: RareLabelEncoder.ipynb

# # RareLabelEncoder
#
# The RareLabelEncoder() groups labels that show a small number of observations in the dataset into a new category called 'Rare'. This helps to avoid overfitting.
#
# The argument ' tol ' indicates the percentage of observations that the label needs to have in order not to be re-grouped into the "Rare" label.<br> The argument n_categories indicates the minimum number of distinct categories that a variable needs to have for any of the labels to be re-grouped into 'Rare'.<br><br>
# #### Note
# If the number of labels is smaller than n_categories, then the encoder will not group the labels for that variable.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import RareLabelEncoder


# Load titanic dataset from file

def load_titanic(filepath='titanic.csv'):
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    return data


data = load_titanic("../data/titanic-2/Titanic-Dataset.csv")
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


# The RareLabelEncoder() groups rare / infrequent categories in
# a new category called "Rare", or any other name entered by the user.
#
# For example in the variable colour,<br> if the percentage of observations
# for the categories magenta, cyan and burgundy 
# are < 5%, all those
# categories will be replaced by the new label "Rare".
#
# Note, infrequent labels can also be grouped under a user defined name, for
# example 'Other'. The name to replace infrequent categories is defined
# with the parameter replace_with.
#
# The encoder will encode only categorical variables (type 'object'). A list
# of variables can be passed as an argument. If no variables are passed as 
# argument, the encoder will find and encode all categorical variables
# (object type).


# Rare value encoder
'''
Parameters
----------

tol: float, default=0.05
    the minimum frequency a label should have to be considered frequent.
    Categories with frequencies lower than tol will be grouped.

n_categories: int, default=10
    the minimum number of categories a variable should have for the encoder
    to find frequent labels. If the variable contains less categories, all
    of them will be considered frequent.

max_n_categories: int, default=None
    the maximum number of categories that should be considered frequent.
    If None, all categories with frequency above the tolerance (tol) will be
    considered.

variables : list, default=None
    The list of categorical variables that will be encoded. If None, the 
    encoder will find and select all object type variables.

replace_with : string, default='Rare'
    The category name that will be used to replace infrequent categories.
'''

rare_encoder = RareLabelEncoder(tol=0.05,
                                n_categories=5,
                                variables=['cabin', 'pclass', 'embarked'])
rare_encoder.fit(X_train)


rare_encoder.encoder_dict_


train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)

test_t.head()


test_t.cabin.value_counts()


# #### The user can change the string from 'Rare' to something else.


# Rare value encoder
rare_encoder = RareLabelEncoder(tol=0.03,
                                replace_with='Other',  # replacing 'Rare' with 'Other'
                                variables=['cabin', 'pclass', 'embarked'],
                                n_categories=2
                                )

rare_encoder.fit(X_train)

train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)

test_t.sample(5)


rare_encoder.encoder_dict_


test_t.cabin.value_counts()


# #### The user can choose to retain only the most popular categories with the argument max_n_categories.


# Rare value encoder

rare_encoder = RareLabelEncoder(tol=0.03,
                                variables=['cabin', 'pclass', 'embarked'],
                                n_categories=2,
                                # keeps only the most popular 3 categories in every variable.
                                max_n_categories=3
                                )

rare_encoder.fit(X_train)

train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)

test_t.sample(5)


rare_encoder.encoder_dict_


# ### Automatically select all categorical variables
#
# If no variable list is passed as argument, it selects all the categorical variables.


len(X_train['pclass'].unique()), len(X_train['sex'].unique()), len(X_train['embarked'].unique())


# # X_train['pclass'].value_counts(dropna=False)
# pclass_encoder = RareLabelEncoder(tol=0.03, n_categories=3)
# X_train['pclass'] = pclass_encoder.fit_transform(X_train[['pclass']])


# # X_train['sex'].value_counts(dropna=False)
# sex_encoder = RareLabelEncoder(tol=0.03, n_categories=2)
# X_train['sex'] = sex_encoder.fit_transform(X_train[['sex']])


# # X_train['embarked'].value_counts(dropna=False)
# embarked_encoder = RareLabelEncoder(tol=0.03, n_categories=3)
# X_train['embarked'] = embarked_encoder.fit_transform(X_train[['embarked']])


## Rare value encoder
rare_encoder = RareLabelEncoder(tol = 0.03, n_categories=3)
rare_encoder.fit(X_train)
rare_encoder.encoder_dict_


train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)
test_t.sample(5)

