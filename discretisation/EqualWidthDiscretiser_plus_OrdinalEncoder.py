import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.discretisation import EqualWidthDiscretiser
from feature_engine.encoding import OrdinalEncoder
plt.rcParams['figure.figsize'] = [15, 5]


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
X = data.drop(['survived'], axis=1)
y = data.survived
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=0)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
X_train[['age', 'fare']].hist(bins=30)
plt.show()
ewd = EqualWidthDiscretiser(bins=5, variables=['age', 'fare'],
    return_object=True)
oe = OrdinalEncoder(variables=['age', 'fare'])
transformer = Pipeline(steps=[('EqualWidthDiscretiser', ewd), (
    'OrdinalEncoder', oe)])
transformer.fit(X_train, y_train)
transformer.named_steps['EqualWidthDiscretiser'].binner_dict_
transformer.named_steps['OrdinalEncoder'].encoder_dict_
train_t = transformer.transform(X_train)
test_t = transformer.transform(X_test)
test_t.head()
plt.figure(figsize=(7, 5))
pd.concat([test_t, y_test], axis=1).groupby('fare')['survived'].mean().plot()
plt.title('Relationship between fare and target')
plt.xlabel('fare')
plt.ylabel('Mean of target')
plt.show()
