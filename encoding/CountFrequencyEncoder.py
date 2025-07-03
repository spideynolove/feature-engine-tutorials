import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.encoding import CountFrequencyEncoder


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

encoding_method : str, default='count' 
                Desired method of encoding.

        'count': number of observations per category
        
        'frequency': percentage of observations per category

variables : list
          The list of categorical variables that will be encoded. If None, the 
          encoder will find and transform all object type variables.
"""
count_encoder = CountFrequencyEncoder(encoding_method='frequency',
    variables=['cabin', 'pclass', 'embarked'])
count_encoder.fit(X_train)
count_encoder.encoder_dict_
train_t = count_encoder.transform(X_train)
test_t = count_encoder.transform(X_test)
test_t.head()
test_t['pclass'].value_counts().plot.bar()
plt.show()
test_orig = count_encoder.inverse_transform(test_t)
test_orig.head()
count_enc = CountFrequencyEncoder(encoding_method='count', variables='cabin')
count_enc.fit(X_train)
count_enc.encoder_dict_
train_t = count_enc.transform(X_train)
test_t = count_enc.transform(X_test)
test_t.head()
count_enc = CountFrequencyEncoder(encoding_method='count')
count_enc.fit(X_train)
count_enc.variables
train_t = count_enc.transform(X_train)
test_t = count_enc.transform(X_test)
test_t.head()
