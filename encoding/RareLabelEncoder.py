import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import RareLabelEncoder


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
"""
rare_encoder = RareLabelEncoder(tol=0.05, n_categories=5, variables=[
    'cabin', 'pclass', 'embarked'])
rare_encoder.fit(X_train)
rare_encoder.encoder_dict_
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)
test_t.head()
test_t.cabin.value_counts()
rare_encoder = RareLabelEncoder(tol=0.03, replace_with='Other', variables=[
    'cabin', 'pclass', 'embarked'], n_categories=2)
rare_encoder.fit(X_train)
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)
test_t.sample(5)
rare_encoder.encoder_dict_
test_t.cabin.value_counts()
rare_encoder = RareLabelEncoder(tol=0.03, variables=['cabin', 'pclass',
    'embarked'], n_categories=2, max_n_categories=3)
rare_encoder.fit(X_train)
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)
test_t.sample(5)
rare_encoder.encoder_dict_
len(X_train['pclass'].unique()), len(X_train['sex'].unique()), len(X_train[
    'embarked'].unique())
rare_encoder = RareLabelEncoder(tol=0.03, n_categories=3)
rare_encoder.fit(X_train)
rare_encoder.encoder_dict_
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_train)
test_t.sample(5)
