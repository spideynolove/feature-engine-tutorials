import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from feature_engine.selection import SelectByTargetMeanPerformance
data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')
data.drop(labels=['name', 'boat', 'ticket', 'body', 'home.dest'], axis=1,
    inplace=True)
data = data.replace('?', np.nan)
data.dropna(subset=['embarked', 'fare'], inplace=True)
data['age'] = data['age'].astype('float')
data['age'] = data['age'].fillna(data['age'].mean())
data['fare'] = data['fare'].astype('float')


def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return 'N'


data['cabin'] = data['cabin'].apply(get_first_cabin)
data.head()
data['cabin'] = data['cabin'].str[0]
data['cabin'] = np.where(data['cabin'].isin(['T', 'G']), 'N', data['cabin'])
data['cabin'].unique()
data.dtypes
data['parch'].value_counts()
data['parch'] = np.where(data['parch'] > 3, 3, data['parch'])
data['sibsp'].value_counts()
data['sibsp'] = np.where(data['sibsp'] > 3, 3, data['sibsp'])
data[['pclass', 'sibsp', 'parch']] = data[['pclass', 'sibsp', 'parch']].astype(
    'O')
data.isnull().sum()
X_train, X_test, y_train, y_test = train_test_split(data.drop(['survived'],
    axis=1), data['survived'], test_size=0.3, random_state=0)
X_train.shape, X_test.shape
sel = SelectByTargetMeanPerformance(variables=None, scoring='roc_auc_score',
    threshold=0.6, bins=3, strategy='equal_frequency', cv=2, random_state=1)
sel.fit(X_train, y_train)
sel.variables_categorical_
sel.variables_numerical_
sel.feature_performance_
sel.features_to_drop_
X_train = sel.transform(X_train)
X_test = sel.transform(X_test)
X_train.shape, X_test.shape
