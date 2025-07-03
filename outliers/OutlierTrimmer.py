import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.outliers import OutlierTrimmer


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
print('train data shape before removing outliers:', X_train.shape)
print('test data shape before removing outliers:', X_test.shape)
print('Max age:', data.age.max())
print('Max fare:', data.fare.max())
print('Min age:', data.age.min())
print('Min fare:', data.fare.min())
plot_hist(data, 'age')
plot_hist(data, 'fare')
"""Parameters
----------

capping_method : str, default=gaussian
    Desired capping method. Can take 'gaussian', 'iqr' or 'quantiles'.
    
tail : str, default=right
    Whether to cap outliers on the right, left or both tails of the distribution.
    Can take 'left', 'right' or 'both'.

fold: int or float, default=3
    How far out to to place the capping values. The number that will multiply
    the std or IQR to calculate the capping values.

variables : list, default=None

missing_values: string, default='raise'
    Indicates if missing values should be ignored or raised."""
trimmer = OutlierTrimmer(capping_method='gaussian', tail='right', fold=3,
    variables=['age', 'fare'])
trimmer.fit(X_train)
trimmer.right_tail_caps_
trimmer.left_tail_caps_
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)
print('Max age:', train_t.age.max())
print('Max fare:', train_t.fare.max())
print('train data shape after removing outliers:', train_t.shape)
print(f'{X_train.shape[0] - train_t.shape[0]} observations are removed\n')
print('test data shape after removing outliers:', test_t.shape)
print(f'{X_test.shape[0] - test_t.shape[0]} observations are removed')
trimmer = OutlierTrimmer(capping_method='gaussian', tail='both', fold=2,
    variables=['fare', 'age'])
trimmer.fit(X_train)
print('Minimum caps :', trimmer.left_tail_caps_)
print('Maximum caps :', trimmer.right_tail_caps_)
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)
print('train data shape after removing outliers:', train_t.shape)
print(f'{X_train.shape[0] - train_t.shape[0]} observations are removed\n')
print('test data shape after removing outliers:', test_t.shape)
print(f'{X_test.shape[0] - test_t.shape[0]} observations are removed')
trimmer = OutlierTrimmer(capping_method='iqr', tail='both', variables=[
    'age', 'fare'])
trimmer.fit(X_train)
print('Minimum caps :', trimmer.left_tail_caps_)
print('Maximum caps :', trimmer.right_tail_caps_)
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)
print('train data shape after removing outliers:', train_t.shape)
print(f'{X_train.shape[0] - train_t.shape[0]} observations are removed\n')
print('test data shape after removing outliers:', test_t.shape)
print(f'{X_test.shape[0] - test_t.shape[0]} observations are removed')
trimmer = OutlierTrimmer(capping_method='quantiles', tail='both', fold=0.02,
    variables=['age', 'fare'])
trimmer.fit(X_train)
print('Minimum caps :', trimmer.left_tail_caps_)
print('Maximum caps :', trimmer.right_tail_caps_)
train_t = trimmer.transform(X_train)
test_t = trimmer.transform(X_test)
print('train data shape after removing outliers:', train_t.shape)
print(f'{X_train.shape[0] - train_t.shape[0]} observations are removed\n')
print('test data shape after removing outliers:', test_t.shape)
print(f'{X_test.shape[0] - test_t.shape[0]} observations are removed')
plot_hist(train_t, 'age')
plot_hist(train_t, 'fare')
