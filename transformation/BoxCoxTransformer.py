# Generated from: BoxCoxTransformer.ipynb
# Warning: This is an auto-generated file. Changes may be overwritten.

# # Variable transformers : BoxCoxTransformer
#
# The BoxCoxTransformer() applies the BoxCox transformation to numerical
# variables.
#
# The Box-Cox transformation is defined as:
#
# - T(Y)=(Y exp(λ)−1)/λ if λ!=0
# - log(Y) otherwise
#
# where Y is the response variable and λ is the transformation parameter. λ varies,
# typically from -5 to 5. In the transformation, all values of λ are considered and
# the optimal value for a given variable is selected.
#
# **For this demonstration, we use the Ames House Prices dataset produced by Professor Dean De Cock:**
#
# Dean De Cock (2011) Ames, Iowa: Alternative to the Boston Housing
# Data as an End of Semester Regression Project, Journal of Statistics Education, Vol.19, No. 3
#
# http://jse.amstat.org/v19n3/decock.pdf
#
# https://www.tandfonline.com/doi/abs/10.1080/10691898.2011.11889627
#
# The version of the dataset used in this notebook can be obtained from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from feature_engine.imputation import ArbitraryNumberImputer, CategoricalImputer
from feature_engine.transformation import BoxCoxTransformer


# #Read data
# data = pd.read_csv('houseprice.csv')
# data.head()

# # let's separate into training and testing set

# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop(['Id', 'SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

# X_train.shape, X_test.shape


# Read the separate files
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')

# Separate features and target in training data
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']

# For test data, you might not have the target variable
X_test = test_df.drop(['Id'], axis=1)  # Note: test data might not have SalePrice column

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# let's transform 2 variables

bct = BoxCoxTransformer(variables = ['LotArea', 'GrLivArea'])

# find the optimal lambdas 
bct.fit(X_train)


# these are the exponents for the BoxCox transformation

bct.lambda_dict_


# transfor the variables

train_t = bct.transform(X_train)
test_t = bct.transform(X_test)


# variable before transformation
X_train['GrLivArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('GrLivArea')


# transformed variable
train_t['GrLivArea'].hist(bins=50)
plt.title('Transformed variable')
plt.xlabel('GrLivArea')


# tvariable before transformation
X_train['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')


# transformed variable
train_t['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')


# ## Automatically select numerical variables
#
# The transformer will transform all numerical variables if no variables are specified.


# load numerical variables only

variables = ['LotFrontage', 'LotArea',
             '1stFlrSF', 'GrLivArea',
             'TotRmsAbvGrd', 'SalePrice']


# data = pd.read_csv('houseprice.csv', usecols=variables)

# # let's separate into training and testing set
# X_train, X_test, y_train, y_test = train_test_split(
#     data.drop(['SalePrice'], axis=1), data['SalePrice'], test_size=0.3, random_state=0)

# X_train.shape, X_test.shape


# Read the separate files
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')

# Separate features and target in training data
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']

# For test data, you might not have the target variable
X_test = test_df.drop(['Id'], axis=1)  # Note: test data might not have SalePrice column

print("X_train :", X_train.shape)
print("X_test :", X_test.shape)


# Impute missing values

arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)

arbitrary_imputer.fit(X_train)

# impute variables
train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)


numeric_columns = train_t.select_dtypes(include=['int64', 'float64']).columns
numeric_columns


train_numeric = train_t[numeric_columns].copy()


for column in numeric_columns:
    min_val = train_numeric[column].min()
    if min_val <= 0:
        print(f"{column}: minimum value = {min_val}")
        shift = abs(min_val) + 1
        train_numeric[column] = train_numeric[column] + shift



train_numeric.describe()


# Check for extremely large values
for col in train_numeric.columns:
    q75 = train_numeric[col].quantile(0.75)
    q25 = train_numeric[col].quantile(0.25)
    iqr = q75 - q25
    upper_bound = q75 + 1.5 * iqr
    
    if train_numeric[col].max() > upper_bound:
        print(f"\n{col}:")
        print(f"Max value: {train_numeric[col].max()}")
        print(f"Upper bound: {upper_bound}")


# # Temp
# from feature_engine.transformation import YeoJohnsonTransformer
# from feature_engine.outliers import Winsorizer

# # First winsorize the outliers
# winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
# train_winsorized = winsor.fit_transform(train_numeric)

# # Then apply YeoJohnson transformation (which handles both positive and negative values)
# yjt = YeoJohnsonTransformer()
# train_transformed = yjt.fit_transform(train_winsorized)

# # Check the results
# print("\nSkewness before transformation:")
# print(train_numeric.skew())
# print("\nSkewness after transformation:")
# print(train_transformed.skew())


# let's transform all numerical variables

bct = BoxCoxTransformer()

# bct.fit(train_t)
bct.fit(train_numeric)


# variables that will be transformed

bct.variables_


# transform  variables
train_t = bct.transform(train_t)
test_t = bct.transform(test_t)


# learned parameters

bct.lambda_dict_

