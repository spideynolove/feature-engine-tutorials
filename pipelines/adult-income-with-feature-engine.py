# Generated from: adult-income-with-feature-engine.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Feature creation within a Pipeline
#
# This notebook shows the creation of new features with Feature-engine and the scikit-learn pipeline.
#
# The notebook uses data from UCI Dataset called "Adult Income"
#
# The data is publicly available on [UCI repository](https://archive.ics.uci.edu/ml/datasets/adult)
#
# Donor:
#
# Ronny Kohavi and Barry Becker
# Data Mining and Visualization
# Silicon Graphics.
# e-mail: ronnyk '@' live.com for questions.


# Import packages


import feature_engine
feature_engine.__version__


# for data processing and visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# for the model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

# for feature engineering
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce


# Load dataset

filename = '../adult.data'
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss', 'hours-per-week', 'native-country','income']
data = pd.read_csv(filename, sep=',', names=col_names)

print(data.shape)

data.head()


# inspect data types

data.info()


# make lists of variable types

categorical = [var for var in data.columns if data[var].dtype == 'O']
discrete = [var for var in data.columns if data[var].dtype != 'O']


categorical


discrete


# histograms of discrete variables

data[discrete].hist(bins=30, figsize=(15,15))
plt.show()


# plot of categoricals

for var in categorical:
    sns.catplot(data=data, y=var, kind="count", palette="ch:.25")


# transform the income value (target variable)


data['income'] = data.income.apply(lambda x: x.replace("<=50K","0"))


data['income'] = data.income.apply(lambda x: x.replace(">50K","1"))


data['income'] = data.income.apply(lambda x: int(x))


data.head()


# split training data into train and test

X_train, X_test, y_train, y_test = train_test_split(data.drop(
    ['income'], axis=1),
    data['income'],
    test_size=0.1,
    random_state=42)

X_train.shape, X_test.shape


# remove 'income' from the categorical list

categorical.pop()


# build the pipeline

income_pipe = Pipeline([

    # === rare label encoding =========
    ('rare_label_enc', ce.RareLabelEncoder(tol=0.1, n_categories=1)),

    # === encoding categories ===
    ('categorical_enc', ce.DecisionTreeEncoder(regression=False,
        param_grid={'max_depth': [1, 2,3]},
        random_state=2909,
        variables=categorical)),

    # === discretisation =====
    ('discretisation', dsc.DecisionTreeDiscretiser(regression=False,
        param_grid={'max_depth': [1, 2, 3]},
        random_state=2909,
        variables=discrete)),

    # classification
    ('gbm', GradientBoostingClassifier(random_state=42))
])


# fit the pipeline

income_pipe.fit(X_train, y_train)


# extract predictions

X_train_preds = income_pipe.predict(X_train)
X_test_preds = income_pipe.predict(X_test)


# show model performance:

print('train accuracy: {}'.format(accuracy_score(y_train, X_train_preds)))
print()
print('test accuracy: {}'.format(accuracy_score(y_test, X_test_preds)))

