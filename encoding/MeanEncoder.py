import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import MeanEncoder


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
variables : list, default=None
    The list of categorical variables that will be encoded. If None, the 
    encoder will find and select all object type variables.
"""
mean_enc = MeanEncoder(variables=['cabin', 'pclass', 'embarked'])
mean_enc.fit(X_train, y_train)
mean_enc.encoder_dict_
train_t = mean_enc.transform(X_train)
test_t = mean_enc.transform(X_test)
test_t.head()
""" The MeanEncoder has the characteristic that return monotonic
 variables, that is, encoded variables which values increase as the target increases"""
plt.figure(figsize=(7, 5))
pd.concat([test_t, y_test], axis=1).groupby('pclass')['survived'].mean().plot()
plt.yticks(np.arange(0, 1.1, 0.1))
plt.title('Relationship between pclass and target')
plt.xlabel('Pclass')
plt.ylabel('Mean of target')
plt.show()
mean_enc = MeanEncoder()
mean_enc.fit(X_train, y_train)
mean_enc.variables
train_t = mean_enc.transform(X_train)
test_t = mean_enc.transform(X_test)
test_t.head()
