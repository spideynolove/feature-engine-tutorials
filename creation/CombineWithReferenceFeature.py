# ## Feature Creation: Combine with reference feature
# The CombineWithReferenceFeature() applies combines a group of variables with a group of reference variables utilising mathematical operations ['sub', 'div','add','mul'], returning one or more additional features as a result.

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

from feature_engine.creation import RelativeFeatures
from feature_engine.creation import MathFeatures

from feature_engine.imputation import MeanMedianImputer

pd.set_option('display.max_columns', None)


# Read data
data = pd.read_csv('winequality-red.csv', sep=';')

data.head()


# **This Data contains 11 features, all numerical, with no missing values.**


# Let's transform the Target, i.e Wine Quality into a binary classification problem:

bins = [0,5,10]

labels = [0, 1] # 'low'=0, 'high'=1

data['quality_range']= pd.cut(x=data['quality'], bins=bins, labels=labels)

data[['quality_range','quality']].head(5)


data.shape


# drop original target

data.drop('quality', axis=1, inplace = True)


data.shape


# ### Sub and Div Combinators:
#
# Let's create two new variables:
#
# - non_free_sulfur_dioxide = total sulfur dioxide - free sulfur dioxide
# - percentage_free_sulfur = free sulfur dioxide / total sulfur dioxide


import operator


def binary_add(x):
    return x.iloc[0] + x.iloc[1]


def binary_sub(x):
    return x.iloc[0] - x.iloc[1]


def binary_div(x):
    return x.iloc[0] / x.iloc[1]


def binary_mul(x):
    return x.iloc[0] * x.iloc[1]



# this transformer substracts free sulfur from total sulfur
sub_with_reference_feature = RelativeFeatures(
    variables=['total sulfur dioxide'],
    reference=['free sulfur dioxide'],
    func=['sub'],
)

# this transformer divides free sulfur by total sulfur
div_with_reference_feature = RelativeFeatures(
    variables=['free sulfur dioxide'],
    reference=['total sulfur dioxide'],
    func=['div'],
)



# # Create the Combinators

# Fit the Sub Combinator on training data
sub_with_reference_feature.fit(data)


# perform the substraction
data_t = sub_with_reference_feature.transform(data)


# perform division
# We can combine both steps in a single call with ".fit_transform()" method
data_t = div_with_reference_feature.fit_transform(data_t)


# Note the additional variables at the end of the dataframe

data_t.head()


# #### Combine with more than 1 operation
# Create the Combinator
multiple_combinator = RelativeFeatures(
    variables=['fixed acidity'],
    reference=['volatile acidity'],
    func=['div', 'add'],
)

# Fit the Combinator to the training data
multiple_combinator.fit(data_t)

# Transform the data
data_t = multiple_combinator.transform(data_t)
# Note the additional variables at the end of the dataframe
data_t.head()


# ### Pipeline Example
# We can put all these transformations into single pipeline:
# Create new variables scale features and train a Logistic Regression model to predict the wine quality range.
# See more on how to use Feature-engine within Scikit-learn Pipelines in these [examples](https://github.com/solegalli/feature_engine/tree/master/examples/Pipelines)

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
    
    # Create new features
    ('subtraction', RelativeFeatures(
        variables=['total sulfur dioxide'],
        reference=['free sulfur dioxide'],
        func=['sub'],
    )
    ),

    ('ratio', RelativeFeatures(
        variables=['free sulfur dioxide'],
        reference=['total sulfur dioxide'],
        func=['div'],
    )
    ),

    ('acidity', RelativeFeatures(
        variables=['fixed acidity'],
        reference=['volatile acidity'],
        func=['div', 'add'],
    )
    ),

    # scale features
    ('scaler', StandardScaler()),

    # Logistic Regression
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