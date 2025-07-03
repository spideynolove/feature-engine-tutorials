import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve,classification_report, confusion_matrix
from sklearn.pipeline import Pipeline as pipe
from sklearn.preprocessing import StandardScaler
from feature_engine.creation import RelativeFeatures
pd.set_option('display.max_columns', None)
data = pd.read_csv('../data/winequality-red.csv', sep=';')
bins = [0, 5, 10]
labels = [0, 1]
data['quality_range'] = pd.cut(x=data['quality'], bins=bins, labels=labels)
data[['quality_range', 'quality']].head(5)
data.shape
data.drop('quality', axis=1, inplace=True)
data.shape


def binary_add(x):
    return x.iloc[0] + x.iloc[1]


def binary_sub(x):
    return x.iloc[0] - x.iloc[1]


def binary_div(x):
    return x.iloc[0] / x.iloc[1]


def binary_mul(x):
    return x.iloc[0] * x.iloc[1]


sub_with_reference_feature = RelativeFeatures(variables=[
    'total sulfur dioxide'], reference=['free sulfur dioxide'], func=['sub'])
div_with_reference_feature = RelativeFeatures(variables=[
    'free sulfur dioxide'], reference=['total sulfur dioxide'], func=['div'])
sub_with_reference_feature.fit(data)
data_t = sub_with_reference_feature.transform(data)
data_t = div_with_reference_feature.fit_transform(data_t)
data_t.head()
multiple_combinator = RelativeFeatures(variables=['fixed acidity'],
    reference=['volatile acidity'], func=['div', 'add'])
multiple_combinator.fit(data_t)
data_t = multiple_combinator.transform(data_t)
data_t.head()
X = data.drop(['quality_range'], axis=1)
y = data.quality_range
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
    random_state=0, shuffle=True, stratify=y)
X_train.shape, X_test.shape
value_pipe = pipe([('subtraction', RelativeFeatures(variables=[
    'total sulfur dioxide'], reference=['free sulfur dioxide'], func=['sub'
    ])), ('ratio', RelativeFeatures(variables=['free sulfur dioxide'],
    reference=['total sulfur dioxide'], func=['div'])), ('acidity',
    RelativeFeatures(variables=['fixed acidity'], reference=[
    'volatile acidity'], func=['div', 'add'])), ('scaler', StandardScaler()
    ), ('LogisticRegression', LogisticRegression())])
value_pipe.fit(X_train, y_train)
pred_train = value_pipe.predict(X_train)
pred_test = value_pipe.predict(X_test)
print('Logistic Regression Model train accuracy score: {}'.format(
    accuracy_score(y_train, pred_train)))
print()
print('Logistic Regression Model test accuracy score: {}'.format(
    accuracy_score(y_test, pred_test)))
print("""Logistic Regression Model test classification report: 

 {}""".
    format(classification_report(y_test, pred_test)))
score = round(accuracy_score(y_test, pred_test), 3)
cm = confusion_matrix(y_test, pred_test)
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Accuracy Score: {0}'.format(score), size=15)
plt.show()
probs = value_pipe.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, probs)
plt.figure(figsize=(8, 5))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate = 1 - Specificity Score')
plt.ylabel('True Positive Rate  = Recall Score')
plt.title('ROC Curve')
plt.show()
