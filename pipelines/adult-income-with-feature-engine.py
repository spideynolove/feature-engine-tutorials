import feature_engine
feature_engine.__version__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce
filename = '../data/adult/adult.data'
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
    'income']
data = pd.read_csv(filename, sep=',', names=col_names)
print(data.shape)
data.head()
data.info()
categorical = [var for var in data.columns if data[var].dtype == 'O']
discrete = [var for var in data.columns if data[var].dtype != 'O']
categorical
discrete
data[discrete].hist(bins=30, figsize=(15, 15))
plt.show()
for var in categorical:
    sns.catplot(data=data, y=var, hue=var, kind='count', palette='ch:.25',
        legend=False)
data['income'] = data.income.apply(lambda x: x.replace('<=50K', '0'))
data['income'] = data.income.apply(lambda x: x.replace('>50K', '1'))
data['income'] = data.income.apply(lambda x: int(x))
data.head()
X_train, X_test, y_train, y_test = train_test_split(data.drop(['income'],
    axis=1), data['income'], test_size=0.1, random_state=42)
X_train.shape, X_test.shape
categorical.pop()
income_pipe = Pipeline([('rare_label_enc', ce.RareLabelEncoder(tol=0.1,
    n_categories=1)), ('categorical_enc', ce.DecisionTreeEncoder(regression
    =False, param_grid={'max_depth': [1, 2, 3]}, random_state=2909,
    variables=categorical)), ('discretisation', dsc.DecisionTreeDiscretiser
    (regression=False, param_grid={'max_depth': [1, 2, 3]}, random_state=
    2909, variables=discrete)), ('gbm', GradientBoostingClassifier(
    random_state=42))])
income_pipe.fit(X_train, y_train)
X_train_preds = income_pipe.predict(X_train)
X_test_preds = income_pipe.predict(X_test)
print('train accuracy: {}'.format(accuracy_score(y_train, X_train_preds)))
print()
print('test accuracy: {}'.format(accuracy_score(y_test, X_test_preds)))
