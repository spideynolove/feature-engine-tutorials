import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.discretisation import DecisionTreeDiscretiser
plt.rcParams['figure.figsize'] = [15, 5]
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['LotArea', 'GrLivArea']].hist(bins=50)
plt.show()
"""
Parameters
----------

cv : int, default=3
    Desired number of cross-validation fold to be used to fit the decision
    tree.

scoring: str, default='neg_mean_squared_error'
    Desired metric to optimise the performance for the tree. Comes from
    sklearn metrics. See DecisionTreeRegressor or DecisionTreeClassifier
    model evaluation documentation for more options:
    https://scikit-learn.org/stable/modules/model_evaluation.html

variables : list
    The list of numerical variables that will be transformed. If None, the
    discretiser will automatically select all numerical type variables.

regression : boolean, default=True
    Indicates whether the discretiser should train a regression or a classification
    decision tree.

param_grid : dictionary, default=None
    The list of parameters over which the decision tree should be optimised
    during the grid search. The param_grid can contain any of the permitted
    parameters for Scikit-learn's DecisionTreeRegressor() or
    DecisionTreeClassifier().

    If None, then param_grid = {'max_depth': [1, 2, 3, 4]}

random_state : int, default=None
    The random_state to initialise the training of the decision tree. It is one
    of the parameters of the Scikit-learn's DecisionTreeRegressor() or
    DecisionTreeClassifier(). For reproducibility it is recommended to set
    the random_state to an integer.
"""
treeDisc = DecisionTreeDiscretiser(cv=3, scoring='neg_mean_squared_error',
    variables=['LotArea', 'GrLivArea'], regression=True, random_state=29)
treeDisc.fit(X_train, y_train)
treeDisc.binner_dict_
train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)
train_t['GrLivArea'].unique()
train_t['LotArea'].unique()
tmp = pd.concat([X_train[['LotArea', 'GrLivArea']], train_t[['LotArea',
    'GrLivArea']]], axis=1)
tmp.columns = ['LotArea', 'GrLivArea', 'LotArea_binned', 'GrLivArea_binned']
tmp.head()
plt.subplot(1, 2, 1)
tmp.groupby('GrLivArea_binned')['GrLivArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of houses per discrete value')
plt.subplot(1, 2, 2)
tmp.groupby('LotArea_binned')['LotArea'].count().plot.bar()
plt.ylabel('Number of houses')
plt.ylabel('Number of houses')
plt.show()


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
X_train, X_test, y_train, y_test = train_test_split(data.drop(['survived'],
    axis=1), data['survived'], test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)
X_train[['fare', 'age']].dtypes
treeDisc = DecisionTreeDiscretiser(cv=3, scoring='roc_auc', variables=[
    'fare', 'age'], regression=False, param_grid={'max_depth': [1, 2]},
    random_state=29)
treeDisc.fit(X_train, y_train)
treeDisc.binner_dict_
train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)
train_t['age'].unique()
train_t['fare'].unique()
tmp = pd.concat([X_train[['fare', 'age']], train_t[['fare', 'age']]], axis=1)
tmp.columns = ['fare', 'age', 'fare_binned', 'age_binned']
tmp.head()
plt.subplot(1, 2, 1)
tmp.groupby('fare_binned')['fare'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of houses per discrete value')
plt.subplot(1, 2, 2)
tmp.groupby('age_binned')['age'].count().plot.bar()
plt.ylabel('Number of houses')
plt.title('Number of houses per discrete value')
plt.show()
pd.concat([test_t, y_test], axis=1).groupby('age')['survived'].mean().plot(
    figsize=(6, 4))
plt.ylabel('Mean of target')
plt.title('Relationship between fare and target')
plt.show()
pd.concat([test_t, y_test], axis=1).groupby('fare')['survived'].mean().plot(
    figsize=(6, 4))
plt.ylabel('Mean of target')
plt.title('Relationship between fare and target')
plt.show()
from sklearn.datasets import load_iris
data = pd.DataFrame(load_iris().data, columns=load_iris().feature_names).join(
    pd.Series(load_iris().target, name='type'))
data.head()
data.type.unique()
X_train, X_test, y_train, y_test = train_test_split(data.drop('type', axis=
    1), data['type'], test_size=0.3, random_state=0)
print(X_train.shape)
print(X_test.shape)
X_train[['sepal length (cm)', 'sepal width (cm)']].dtypes
treeDisc = DecisionTreeDiscretiser(cv=3, scoring='accuracy', variables=[
    'sepal length (cm)', 'sepal width (cm)'], regression=False, random_state=29
    )
treeDisc.fit(X_train, y_train)
treeDisc.binner_dict_
train_t = treeDisc.transform(X_train)
test_t = treeDisc.transform(X_test)
tmp = pd.concat([X_train[['sepal length (cm)', 'sepal width (cm)']],
    train_t[['sepal length (cm)', 'sepal width (cm)']]], axis=1)
tmp.columns = ['sepal length (cm)', 'sepal width (cm)', 'sepalLen_binned',
    'sepalWid_binned']
tmp.head()
plt.subplot(1, 2, 1)
tmp.groupby('sepalLen_binned')['sepal length (cm)'].count().plot.bar()
plt.ylabel('Number of species')
plt.title('Number of observations per discrete value')
plt.subplot(1, 2, 2)
tmp.groupby('sepalWid_binned')['sepal width (cm)'].count().plot.bar()
plt.ylabel('Number of species')
plt.title('Number of observations per discrete value')
plt.show()
