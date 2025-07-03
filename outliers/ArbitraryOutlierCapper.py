import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.outliers import ArbitraryOutlierCapper


def load_titanic(filepath='../data/titanic.csv'):
    data = pd.read_csv(filepath)
    data = data.replace('?', np.nan)
    data['cabin'] = data['cabin'].astype(str).str[0]
    data['pclass'] = data['pclass'].astype('O')
    data['embarked'].fillna('C', inplace=True)
    data['fare'] = data['fare'].astype('float')
    data['fare'].fillna(data['fare'].median(), inplace=True)
    data['age'] = data['age'].astype('float')
    data['age'].fillna(data['age'].median(), inplace=True)
    data.drop(['name', 'ticket'], axis=1, inplace=True)
    return data


def plot_hist(data, col):
    plt.figure(figsize=(8, 5))
    plt.hist(data[col], bins=30)
    plt.title('Distribution of ' + col)
    return plt.show()


data = load_titanic()
data.sample(5)
X_train, X_test, y_train, y_test = train_test_split(data.drop('survived',
    axis=1), data['survived'], test_size=0.3, random_state=0)
print('train data:', X_train.shape)
print('test data:', X_test.shape)
plot_hist(data, 'age')
plot_hist(data, 'fare')
print('Max age:', data.age.max())
print('Max fare:', data.fare.max())
print('Min age:', data.age.min())
print('Min fare:', data.fare.min())
"""Parameters
----------
max_capping_dict : dictionary, default=None
    Dictionary containing the user specified capping values for the right tail of
    the distribution of each variable (maximum values).

min_capping_dict : dictionary, default=None
    Dictionary containing user specified capping values for the eft tail of the
    distribution of each variable (minimum values).

missing_values : string, default='raise'
    Indicates if missing values should be ignored or raised. If
    `missing_values='raise'` the transformer will return an error if the
    training or the datasets to transform contain missing values.
"""
capper = ArbitraryOutlierCapper(max_capping_dict={'age': 50, 'fare': 150},
    min_capping_dict=None)
capper.fit(X_train)
print('Maximum caps:', capper.right_tail_caps_)
capper.left_tail_caps_
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)
print('Max age after capping:', train_t.age.max())
print('Max fare after capping:', train_t.fare.max())
capper = ArbitraryOutlierCapper(max_capping_dict=None, min_capping_dict={
    'age': 10, 'fare': 100})
capper.fit(X_train)
capper.right_tail_caps_
capper.left_tail_caps_
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)
print('Min age:', train_t.age.min())
print('Min fare:', train_t.fare.min())
capper = ArbitraryOutlierCapper(min_capping_dict={'age': 5, 'fare': 5},
    max_capping_dict={'age': 60, 'fare': 150})
capper.fit(X_train)
capper.right_tail_caps_
capper.left_tail_caps_
train_t = capper.transform(X_train)
test_t = capper.transform(X_test)
print('Max age:', train_t.age.max())
print('Max fare:', train_t.fare.max())
print('Min age:', train_t.age.min())
print('Min fare:', train_t.fare.min())
plot_hist(train_t, 'age')
plot_hist(train_t, 'fare')
