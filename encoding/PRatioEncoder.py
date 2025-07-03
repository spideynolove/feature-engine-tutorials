import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import PRatioEncoder
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
rare_encoder = RareLabelEncoder(tol=0.03, n_categories=2, variables=[
    'cabin', 'pclass', 'embarked'])
rare_encoder.fit(X_train)
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_test)
"""
Parameters
----------

encoding_method : str, default=woe
    Desired method of encoding.

    'ratio' : probability ratio

    'log_ratio' : log probability ratio

variables : list, default=None
    The list of categorical variables that will be encoded. If None, the
    encoder will find and select all object type variables.
"""
Ratio_enc = PRatioEncoder(encoding_method='ratio', variables=['cabin',
    'pclass', 'embarked'])
Ratio_enc.fit(train_t, y_train)
Ratio_enc.encoder_dict_
train_t = Ratio_enc.transform(train_t)
test_t = Ratio_enc.transform(test_t)
test_t.sample(5)
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_test)
logRatio_enc = PRatioEncoder(encoding_method='log_ratio', variables=[
    'cabin', 'pclass', 'embarked'])
logRatio_enc.fit(train_t, y_train)
logRatio_enc.encoder_dict_
train_t = logRatio_enc.transform(train_t)
test_t = logRatio_enc.transform(test_t)
test_t.sample(5)
""" The PRatioEncoder(encoding_method='ratio' or 'log_ratio') has the characteristic that return monotonic
 variables, that is, encoded variables which values increase as the target increases"""
plt.figure(figsize=(7, 5))
pd.concat([test_t, y_test], axis=1).groupby('pclass')['survived'].mean().plot()
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Relationship between pclass and target')
plt.xlabel('Pclass')
plt.ylabel('Mean of target')
plt.show()
train_t = rare_encoder.transform(X_train)
test_t = rare_encoder.transform(X_test)
logRatio_enc = PRatioEncoder(encoding_method='log_ratio')
logRatio_enc.fit(train_t, y_train)
train_t = logRatio_enc.transform(train_t)
test_t = logRatio_enc.transform(test_t)
test_t.sample(5)
