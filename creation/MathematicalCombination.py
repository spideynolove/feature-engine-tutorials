# Generated from: MathematicalCombination.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# ### Feature Creation: MathematicalCombination
# The MathematicalCombination() applies basic mathematical operations **[‘sum’, ‘prod’, ‘mean’, ‘std’, ‘max’, ‘min’]** to multiple features, returning one or more additional features as a result.
#
# For this demonstration, we use the UCI Wine Quality Dataset.
#
# The data is publicly available on **[UCI repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)**
#
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline as pipe
from sklearn.preprocessing import StandardScaler

from feature_engine.creation import MathFeatures
from feature_engine.imputation import MeanMedianImputer

pd.set_option('display.max_columns', None)


# Read data
data = pd.read_csv('../data/winequality-red.csv', sep=';')

data.head()


# **This Data contains 11 features, all numerical, with no missing values.**


# Let's transform the Target, i.e Wine Quality into a binary classification problem:

bins = [0,5,10]

labels = [0, 1] # 'low'=0, 'high'=1

data['quality_range']= pd.cut(x=data['quality'], bins=bins, labels=labels)

data[['quality_range','quality']].head(5)


# drop original target

data.drop('quality', axis=1, inplace = True) 


# ### Sum and Mean Combinators:
# Let's create two new variables:
# - avg_acidity = mean(fixed acidity, volatile acidity)
# - total_minerals = sum(Total sulfure dioxide, sulphates)


# Create the Combinators

math_combinator_mean = MathFeatures(
    variables=['fixed acidity', 'volatile acidity'],
    func = ['mean'],
    new_variables_names = ['avg_acidity']
)

math_combinator_sum = MathFeatures(
    variables=['total sulfur dioxide', 'sulphates'],
    func = ['sum'],
    new_variables_names = ['total_minerals']
)

# Fit the Mean Combinator on training data
math_combinator_mean.fit(data)

# Transform the data
data_t = math_combinator_mean.transform(data)

# We can combine both steps in a single call with ".fit_transform()" methode
data_t = math_combinator_sum.fit_transform(data_t)


data_t.head()


# You can check the mappings between each new variable and the operation it's created with in the **combination_dict_**


# math_combinator_mean.feature_names_in_


math_combinator_mean.variables_


# ### Combine with more than 1 operation
#
# We can also combine the variables with more than 1 mathematical operation. And the transformer has the option to create variable names automatically.


# Create the Combinators

multiple_combinator = MathFeatures(
    variables=['fixed acidity', 'volatile acidity'],
    func = ['mean', 'sum'],
    new_variables_names = None
)


# Fit the Combinator to the training data
multiple_combinator.fit(data)

# Transform the data
data_t = multiple_combinator.transform(data)


# Note the 2 additional variables at the end of the dataframe
data_t.head()


multiple_combinator._get_new_features_name()


# # and here the variable names and the operation that was
# # applied to create that variable

# multiple_combinator.combination_dict_

# # {'mean(fixed acidity-volatile acidity)': 'mean',
# #  'sum(fixed acidity-volatile acidity)': 'sum'}



# ### Pipeline Example


# We can put all these transformations into single pipeline:
#
# 1. Create new variables
# 2. Scale features
# 3. Train a Logistic Regression model to predict wine quality
#
# See more on how to use Feature-engine within Scikit-learn Pipelines in these **[examples](https://github.com/solegalli/feature_engine/tree/master/examples/Pipelines)**


X = data.drop(['quality_range'], axis=1)

y = data.quality_range

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    shuffle=True,
                                                    stratify=y
                                                    )
X_train.shape, X_test.shape


value_pipe = pipe([
    ('math_combinator_mean', MathFeatures(variables=['fixed acidity', 'volatile acidity'],
                                          func=['mean'],
                                          new_variables_names=['avg_acidity'])),
    ('math_combinator_sum', MathFeatures(variables=['total sulfur dioxide', 'sulphates'],
                                         func=['sum'],
                                         new_variables_names=['total_minerals'])),
    ('scaler', StandardScaler()),
    ('LogisticRegression', LogisticRegression())
])


value_pipe.fit(X_train, y_train)


pred_train = value_pipe.predict(X_train)
pred_test = value_pipe.predict(X_test)


print('Logistic Regression Model train accuracy score: {}'.format(
    accuracy_score(y_train, pred_train)))
print()
print('Logistic Regression Model test accuracy score: {}'.format(
    accuracy_score(y_test, pred_test)))


print('Logistic Regression Model test classification report: \n\n {}'.format(
    classification_report(y_test, pred_test)))


score = round(accuracy_score(y_test, pred_test), 3)
cm = confusion_matrix(y_test, pred_test)

sns.heatmap(cm, annot=True, fmt=".0f")
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size=15)
plt.show()


# Predict probabilities for the test data
probs = value_pipe.predict_proba(X_test)[:, 1]

# Get the ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, probs)

# Plot ROC curve
plt.figure(figsize=(8, 5))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate = 1 - Specificity Score')
plt.ylabel('True Positive Rate  = Recall Score')
plt.title('ROC Curve')
plt.show()

