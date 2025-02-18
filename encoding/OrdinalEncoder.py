# Generated from: OrdinalEncoder.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # OrdinalEncoder
# The OrdinalEncoder() will replace the variable labels by digits, from 1 to the number of different labels. 
#
# If we select "arbitrary", then the encoder will assign numbers as the labels appear in the variable (first come first served).
#
# If we select "ordered", the encoder will assign numbers following the mean of the target value for that label. So labels for which the mean of the target is higher will get the number 1, and those where the mean of the target is smallest will get the number n.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from feature_engine.encoding import OrdinalEncoder


# Load titanic dataset from OpenML

def load_titanic(filepath='titanic.csv'):
    # data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    # data.drop(labels=['boat', 'body', 'home.dest', 'name', 'ticket'], axis=1, inplace=True)
    return data


# data = load_titanic("../data/titanic.csv")
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


# The OrdinalEncoder() replaces categories by ordinal numbers 
# (0, 1, 2, 3, etc). The numbers can be ordered based on the mean of the target
# per category, or assigned arbitrarily.
#
# Ordered ordinal encoding:  for the variable colour, if the mean of the target
# for blue, red and grey is 0.5, 0.8 and 0.1 respectively, blue is replaced by 1,
# red by 2 and grey by 0.
#
# Arbitrary ordinal encoding: the numbers will be assigned arbitrarily to the
# categories, on a first seen first served basis.
#
# The encoder will encode only categorical variables (type 'object'). A list
# of variables can be passed as an argument. If no variables are passed, the
# encoder will find and encode all categorical variables (type 'object').


# ### Ordered


# we will encode 3 variables:
'''
Parameters
----------

encoding_method : str, default='ordered' 
    Desired method of encoding.

    'ordered': the categories are numbered in ascending order according to
    the target mean value per category.

    'arbitrary' : categories are numbered arbitrarily.
    
variables : list, default=None
    The list of categorical variables that will be encoded. If None, the 
    encoder will find and select all object type variables.
'''
ordinal_enc = OrdinalEncoder(encoding_method='ordered',
                             variables=['pclass', 'cabin', 'embarked'])

# for this encoder, we need to pass the target as argument
# if encoding_method='ordered'
ordinal_enc.fit(X_train, y_train)


ordinal_enc.encoder_dict_


# transform and visualise the data

train_t = ordinal_enc.transform(X_train)
test_t = ordinal_enc.transform(X_test)

test_t.sample(5)


''' The OrdinalEncoder with encoding_method='order' has the characteristic that return monotonic
 variables,that is, encoded variables which values increase as the target increases'''

# let's explore the monotonic relationship
plt.figure(figsize=(7,5))
pd.concat([test_t,y_test], axis=1).groupby("pclass")["survived"].mean().plot()
plt.xticks([0,1,2])
plt.yticks(np.arange(0,1.1,0.1))
plt.title("Relationship between pclass and target")
plt.xlabel("Pclass")
plt.ylabel("Mean of target")
plt.show()


# ### Arbitrary


ordinal_enc = OrdinalEncoder(encoding_method='arbitrary',
                             variables=['pclass', 'cabin', 'embarked'])

# for this encoder we don't need to add the target. You can leave it or remove it.
ordinal_enc.fit(X_train)


ordinal_enc.encoder_dict_


# Note that the ordering of the different labels is  not the same when we select "arbitrary" or "ordered"


# transform: see the numerical values in the former categorical variables

train_t = ordinal_enc.transform(X_train)
test_t = ordinal_enc.transform(X_test)

test_t.sample(5)


# ### Automatically select categorical variables
#
# This encoder selects all the categorical variables, if None is passed to the variable argument when calling the encoder.


ordinal_enc = OrdinalEncoder(encoding_method = 'arbitrary')

# for this encoder we don't need to add the target. You can leave it or remove it.
ordinal_enc.fit(X_train)


ordinal_enc.variables


train_t = ordinal_enc.transform(X_train)
test_t = ordinal_enc.transform(X_test)

test_t.sample(5)

