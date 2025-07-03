import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import OneHotEncoder


def load_titanic(filepath='titanic.csv'):
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['age'] = data['age'].astype('float').fillna(data.age.median())
    data['fare'] = data['fare'].astype('float').fillna(data.fare.median())
    data['embarked'].fillna('C', inplace=True)
    return data


data = load_titanic('../data/titanic-2/Titanic-Dataset.csv')
data.head()
X = data.drop(['survived', 'name', 'ticket'], axis=1)
y = data.survived
X[['cabin', 'pclass', 'embarked']].isnull().sum()
""" Make sure that the variables are type (object).
if not, cast it as object , otherwise the transformer will either send an error (if we pass it as argument) 
or not pick it up (if we leave variables=None). """
X[['cabin', 'pclass', 'embarked']].dtypes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=0)
X_train.shape, X_test.shape
"""
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
"""
ohe_enc = OneHotEncoder(top_categories=None, variables=['pclass', 'cabin',
    'embarked'], drop_last=False)
ohe_enc.fit(X_train)
ohe_enc.encoder_dict_
train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)
test_t.head()
ohe_enc = OneHotEncoder(top_categories=2, variables=['pclass', 'cabin',
    'embarked'], drop_last=False)
ohe_enc.fit(X_train)
ohe_enc.encoder_dict_
train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)
test_t.head()
ohe_enc = OneHotEncoder(top_categories=None, variables=['pclass', 'cabin',
    'embarked'], drop_last=True)
ohe_enc.fit(X_train)
ohe_enc.encoder_dict_
train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)
test_t.head()
ohe_enc = OneHotEncoder(top_categories=None, drop_last=True)
ohe_enc.fit(X_train)
ohe_enc.variables
ohe_enc.variables_
ohe_enc.variables_binary_
train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)
test_t.head()
ohe_enc = OneHotEncoder(top_categories=None, drop_last=False,
    drop_last_binary=True)
ohe_enc.fit(X_train)
ohe_enc.encoder_dict_
ohe_enc.variables_binary_
train_t = ohe_enc.transform(X_train)
test_t = ohe_enc.transform(X_train)
test_t.head()
