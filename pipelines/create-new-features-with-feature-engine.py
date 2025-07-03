import feature_engine
feature_engine.__version__
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from feature_engine.creation import RelativeFeatures, MathFeatures
data = pd.read_csv('../data/winequality-red.csv', sep=';')
print(data.shape)
data.head()
(data['quality'].value_counts() / len(data)).sort_index().plot.bar()
plt.title('Wine Quality')
plt.ylabel('Percentage of wines in the data')
plt.xlabel('Wine Quality')
plt.show()
data['quality'] = np.where(data['quality'] <= 6, 0, 1)
(data['quality'].value_counts() / len(data)).plot.bar()
plt.title('Wine Quality')
plt.ylabel('Percentage of wines in the data')
plt.xlabel('Wine Quality')
plt.show()
data.hist(bins=50, figsize=(10, 10))
plt.show()
g = sns.PairGrid(data, x_vars=['quality'], y_vars=data.columns[0:-1])
g.map(sns.barplot)
plt.show()
df = data.melt(id_vars=['quality'])
cols = df.variable.unique()
g = sns.axisgrid.FacetGrid(df[df.variable.isin(cols[0:6])], col='variable',
    sharey=False)
g.map(sns.boxplot, 'quality', 'value')
plt.show()
g = sns.axisgrid.FacetGrid(df[df.variable.isin(cols[6:])], col='variable',
    sharey=False)
g.map(sns.boxplot, 'quality', 'value')
plt.show()
data.head()
plt.scatter(data['citric acid'], data['pH'], c=data['quality'])
plt.xlabel('Citric acid')
plt.ylabel('pH')
plt.show()
plt.scatter(data['sulphates'], data['pH'], c=data['quality'])
plt.xlabel('sulphates')
plt.ylabel('pH')
plt.show()
plt.scatter(data['sulphates'], data['citric acid'], c=data['quality'])
plt.xlabel('sulphates')
plt.ylabel('citric acid')
plt.show()
g = sns.PairGrid(data, y_vars=['density'], x_vars=['chlorides', 'sulphates',
    'residual sugar', 'alcohol'])
g.map(sns.regplot)
plt.show()
combinator = MathFeatures(variables=['fixed acidity', 'volatile acidity'],
    func=['sum', 'mean'], new_variables_names=['total_acidity',
    'average_acidity'])
data = combinator.fit_transform(data)
data.head()
combinator = MathFeatures(variables=['chlorides', 'sulphates'], func=['sum',
    'mean'], new_variables_names=['total_minerals', 'average_minerals'])
data = combinator.fit_transform(data)
data.head()
combinator = RelativeFeatures(variables=['total sulfur dioxide'], reference
    =['free sulfur dioxide'], func=['sub'])
data = combinator.fit_transform(data)
data.head()
combinator = RelativeFeatures(variables=['free sulfur dioxide'], reference=
    ['total sulfur dioxide'], func=['div'])
data = combinator.fit_transform(data)
data.head()
combinator = RelativeFeatures(variables=['sulphates'], reference=[
    'free sulfur dioxide'], func=['div'])
data = combinator.fit_transform(data)
data.head()
data.columns
new_vars = ['total_acidity', 'average_acidity', 'total_minerals',
    'average_minerals', 'total sulfur dioxide_sub_free sulfur dioxide',
    'free sulfur dioxide_div_total sulfur dioxide',
    'free sulfur dioxide_div_total sulfur dioxide']
df = data[new_vars + ['quality']].melt(id_vars=['quality'])
cols = df.variable.unique()
g = sns.axisgrid.FacetGrid(df[df.variable.isin(cols)], col='variable',
    sharey=False)
g.map(sns.boxplot, 'quality', 'value')
plt.show()
data = pd.read_csv('../data/winequality-red.csv', sep=';')
data['quality'] = np.where(data['quality'] <= 6, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(data.drop(labels=[
    'quality'], axis=1), data['quality'], test_size=0.2, random_state=0)
X_train.shape, X_test.shape
pipe = Pipeline([('acidity', MathFeatures(variables=['fixed acidity',
    'volatile acidity'], func=['sum', 'mean'], new_variables_names=[
    'total_acidity', 'average_acidity'])), ('total_minerals', MathFeatures(
    variables=['chlorides', 'sulphates'], func=['sum', 'mean'],
    new_variables_names=['total_minerals', 'average_minearals'])), (
    'non_free_sulfur', RelativeFeatures(variables=['total sulfur dioxide'],
    reference=['free sulfur dioxide'], func=['sub'])), ('perc_free_sulfur',
    RelativeFeatures(variables=['free sulfur dioxide'], reference=[
    'total sulfur dioxide'], func=['div'])), ('perc_salt_sulfur',
    RelativeFeatures(variables=['sulphates'], reference=[
    'free sulfur dioxide'], func=['div'])), ('gbm',
    GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=1))])
pipe.fit(X_train, y_train)
pred = pipe.predict_proba(X_train)
print('Train roc-auc: {}'.format(roc_auc_score(y_train, pred[:, 1])))
pred = pipe.predict_proba(X_test)
print('Test roc-auc: {}'.format(roc_auc_score(y_test, pred[:, 1])))
new_vars = ['total_acidity', 'average_acidity', 'total_minerals',
    'average_minearals', 'non_free_sulfur_dioxide',
    'percentage_free_sulfur', 'percentage_salt_sulfur']
importance = pd.Series(pipe.named_steps['gbm'].feature_importances_)
importance.index = list(X_train.columns) + new_vars
importance.sort_values(ascending=False).plot.bar(figsize=(15, 5))
plt.ylabel('Feature importance')
plt.show()
