# Generated from: create-new-features-with-feature-engine.ipynb

# # Create new features within a Pipeline
#
# In this notebook, I show how easy and practical is to create new features Feature-engine and the scikit-learn pipeline.
#
# For this demonstration, we use the UCI Wine Quality Dataset.
#
# The data is publicly available on [UCI repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
#
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.


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


# import classes from Feature-engine
from feature_engine.creation import RelativeFeatures, MathFeatures


# Load dataset

data = pd.read_csv('../data/winequality-red.csv', sep=';')

print(data.shape)

data.head()


# ## Exploratory Data Analysis
#
# Let's have a look at the variables and their relationships.


# check how many wines of different qualities there are

# percentage of wines of each quality
(data['quality'].value_counts() / len(data)).sort_index().plot.bar()

# plot
plt.title('Wine Quality')
plt.ylabel('Percentage of wines in the data')
plt.xlabel('Wine Quality')
plt.show()


# Most wines are medium to low quality. Only a few of high quality (>6)


# let's transform the target into binary

# wines with quality below 6 will be considered low quality (0)
data['quality'] = np.where(data['quality'] <= 6, 0, 1)

(data['quality'].value_counts() / len(data)).plot.bar()

plt.title('Wine Quality')
plt.ylabel('Percentage of wines in the data')
plt.xlabel('Wine Quality')
plt.show()


# let's explore variable distributions with histograms

data.hist(bins=50, figsize=(10,10))

plt.show()


# All variables are continuous.


# let's evaluate the mean variable value per wine quality

g = sns.PairGrid(data, x_vars=["quality"], y_vars=data.columns[0:-1])
g.map(sns.barplot)
plt.show()


# There doesn't seem to be a difference in pH between wines of low and high quality, but high quality wines tend to have more alcohol, for example.
#
# Similarly, good quality wines tend to have more sulphates but less free and total sulfur, a molecule that is part of the sulphates.
#
# Good quality wines tend to have more citric acid, yet surprisingly, the pH in good quality wines is not lower. So the pH must be equilibrated through something else, for example the sulphates.


# now let's explore the data with boxplots

# reorganise for plotting
df = data.melt(id_vars=['quality'])

# capture variables
cols = df.variable.unique()

# plot first 6 columns
g = sns.axisgrid.FacetGrid(df[df.variable.isin(cols[0:6])], col='variable', sharey=False)
g.map(sns.boxplot, 'quality','value')
plt.show()


# plot remaining columns
g = sns.axisgrid.FacetGrid(df[df.variable.isin(cols[6:])], col='variable', sharey=False)
g.map(sns.boxplot, 'quality','value')
plt.show()


data.head()


# the citric acid affects the pH of the wine

plt.scatter(data['citric acid'], data['pH'], c=data['quality'])
plt.xlabel('Citric acid')
plt.ylabel('pH')
plt.show()


# the sulphates may affect the pH of the wine

plt.scatter(data['sulphates'], data['pH'], c=data['quality'])
plt.xlabel('sulphates')
plt.ylabel('pH')
plt.show()


plt.scatter(data['sulphates'], data['citric acid'], c=data['quality'])
plt.xlabel('sulphates')
plt.ylabel('citric acid')
plt.show()


# Good quality wine tend to have more citric acid and more sulphate, thus similar pH.


# let's evaluate the relationship between some molecules and the density of the wine

g = sns.PairGrid(data, y_vars=["density"], x_vars=['chlorides','sulphates', 'residual sugar', 'alcohol'])
g.map(sns.regplot)
plt.show()


# ## Create additional variables
#
# Let's combine variables into new ones to capture additional information.


# combine fixed and volatile acidity to create total acidity
# and mean acidity

combinator = MathFeatures(
    variables=['fixed acidity', 'volatile acidity'],
    func = ['sum', 'mean'],
    new_variables_names = ['total_acidity', 'average_acidity']
)

data = combinator.fit_transform(data)

# note the new variables at the end of the dataframe
data.head()


# let's combine salts into total minerals and average minerals

combinator = MathFeatures(
    variables=['chlorides', 'sulphates'],
    func = ['sum', 'mean'],
    new_variables_names = ['total_minerals', 'average_minerals']
)

data = combinator.fit_transform(data)

# note the new variable at the end of the dataframe
data.head()


# let's determine the sulfur that is not free

combinator = RelativeFeatures(
    variables=['total sulfur dioxide'],
    reference=['free sulfur dioxide'],
    func=['sub'],
    # new_variables_names=['non_free_sulfur_dioxide']
)

data = combinator.fit_transform(data)

# note the new variable at the end of the dataframe
data.head()


# let's calculate the % of free sulfur

combinator = RelativeFeatures(
    variables=['free sulfur dioxide'],
    reference=['total sulfur dioxide'],
    func=['div'],
    # new_variables_names=['percentage_free_sulfur']
)

data = combinator.fit_transform(data)

# note the new variable at the end of the dataframe
data.head()


# let's determine from all free sulfur how much is as salt

combinator = RelativeFeatures(
    variables=['sulphates'],
    reference=['free sulfur dioxide'],
    func=['div'],
    # new_variables_names=['percentage_salt_sulfur']
)

data = combinator.fit_transform(data)

# note the new variable at the end of the dataframe
data.head()


data.columns


# now let's explore the new variables with boxplots

new_vars = [
    'total_acidity',
    'average_acidity',
    'total_minerals',
    'average_minerals',
    
    'total sulfur dioxide_sub_free sulfur dioxide',
    'free sulfur dioxide_div_total sulfur dioxide',
    'free sulfur dioxide_div_total sulfur dioxide',

    # 'non_free_sulfur_dioxide',
    # 'percentage_free_sulfur',
    # 'percentage_salt_sulfur'
]

# KeyError: "['non_free_sulfur_dioxide', 'percentage_free_sulfur', 
# 'percentage_salt_sulfur'] not in index"

# reorganise for plotting
df = data[new_vars+['quality']].melt(id_vars=['quality'])

# capture variables
cols = df.variable.unique()

# plot first 6 columns
g = sns.axisgrid.FacetGrid(df[df.variable.isin(cols)], col='variable', sharey=False)
g.map(sns.boxplot, 'quality','value')
plt.show()


# ## Machine Learning Pipeline
#
# Now we are going to carry out all variable creation within a Scikit-learn Pipeline and add a classifier at the end.


data = pd.read_csv('../data/winequality-red.csv', sep=';')

# make binary target
data['quality'] = np.where(data['quality'] <= 6, 0, 1)

# separate dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['quality'], axis=1),
    data['quality'],
    test_size=0.2,
    random_state=0)

X_train.shape, X_test.shape


pipe = Pipeline([
    # variable creation
    ('acidity', MathFeatures(
        variables=['fixed acidity', 'volatile acidity'],
        func = ['sum', 'mean'],
        new_variables_names = ['total_acidity', 'average_acidity']
        )
    ),
    
    ('total_minerals', MathFeatures(
        variables=['chlorides', 'sulphates'],
        func = ['sum', 'mean'],
        new_variables_names = ['total_minerals', 'average_minearals'],
        )
    ),
    
    ('non_free_sulfur', RelativeFeatures(
        variables=['total sulfur dioxide'],
        reference=['free sulfur dioxide'],
        func=['sub'],
        # new_variables_names=['non_free_sulfur_dioxide'],
        )
    ),
    
    ('perc_free_sulfur', RelativeFeatures(
        variables=['free sulfur dioxide'],
        reference=['total sulfur dioxide'],
        func=['div'],
        # new_variables_names=['percentage_free_sulfur'],
        )
    ),
    
    ('perc_salt_sulfur', RelativeFeatures(
        variables=['sulphates'],
        reference=['free sulfur dioxide'],
        func=['div'],
        # new_variables_names=['percentage_salt_sulfur'],
        )
    ),
    
    # =====  the machine learning model ====
    
    ('gbm', GradientBoostingClassifier(n_estimators=10, max_depth=2, random_state=1)),
])

# create new variables, and then train gradient boosting machine
# uses only the training dataset

pipe.fit(X_train, y_train)


# make predictions and determine model performance

# the pipeline takes in the raw data, creates all the new features and then
# makes the prediction with the model trained on the final subset of variables

# obtain predictions and determine model performance

pred = pipe.predict_proba(X_train)
print('Train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))

pred = pipe.predict_proba(X_test)
print('Test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# ## Feature importance


new_vars = ['total_acidity', 'average_acidity', 'total_minerals', 'average_minearals',
           'non_free_sulfur_dioxide', 'percentage_free_sulfur','percentage_salt_sulfur']


importance = pd.Series(pipe.named_steps['gbm'].feature_importances_)
importance.index = list(X_train.columns) + new_vars

importance.sort_values(ascending=False).plot.bar(figsize=(15,5))
plt.ylabel('Feature importance')
plt.show()


# We see that some of the variables that we created are somewhat important for the prediction, like average_minerals, total_minerals, and total and average acidity.
#
# That is all folks!
#
#
# ## References and further reading
#
# - [Feature-engine](https://feature-engine.readthedocs.io/en/latest/index.html), Python open-source library
# - [Python Feature Engineering Cookbook](https://www.packtpub.com/data/python-feature-engineering-cookbook)
#
# ## Other Kaggle kernels featuring Feature-engine
#
# - [Feature selection for bank customer satisfaction prediction](https://www.kaggle.com/solegalli/feature-selection-with-feature-engine)
# - [Feature engineering and selection for house price prediction](https://www.kaggle.com/solegalli/predict-house-price-with-feature-engine)

