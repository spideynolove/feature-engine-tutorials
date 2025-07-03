import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, log_loss
from sklearn.preprocessing import StandardScaler
from feature_engine import imputation as mdi
from feature_engine import discretisation as dsc
from feature_engine import encoding as ce
data = pd.read_csv('../data/house-prices/train.csv')
categorical = [var for var in data.columns if data[var].dtype == 'O']
year_vars = [var for var in data.columns if 'Yr' in var or 'Year' in var]
discrete = [var for var in data.columns if data[var].dtype != 'O' and len(
    data[var].unique()) < 20 and var not in year_vars]
numerical = [var for var in data.columns if data[var].dtype != 'O' if var
     not in discrete and var not in ['Id', 'SalePrice'] and var not in
    year_vars]
sns.pairplot(data=data, y_vars=['SalePrice'], x_vars=['LotFrontage',
    'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2'])
sns.pairplot(data=data, y_vars=['SalePrice'], x_vars=['BsmtUnfSF',
    'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF'])
sns.pairplot(data=data, y_vars=['SalePrice'], x_vars=['GrLivArea',
    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch'])
sns.pairplot(data=data, y_vars=['SalePrice'], x_vars=['3SsnPorch',
    'ScreenPorch', 'MiscVal'])
data[discrete] = data[discrete].astype('O')
X_train, X_test, y_train, y_test = train_test_split(data.drop(['Id',
    'SalePrice'], axis=1), data['SalePrice'], test_size=0.1, random_state=0)
X_train.shape, X_test.shape


def elapsed_years(df, var):
    df[var] = df['YrSold'] - df[var]
    return df


for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)
X_train.drop('YrSold', axis=1, inplace=True)
X_test.drop('YrSold', axis=1, inplace=True)
house_pipe = Pipeline([('missing_ind', mdi.AddMissingIndicator(missing_only
    =True)), ('imputer_num', mdi.MeanMedianImputer(imputation_method=
    'median')), ('imputer_cat', mdi.CategoricalImputer(return_object=True)),
    ('rare_label_enc', ce.RareLabelEncoder(tol=0.1, n_categories=1)), (
    'categorical_enc', ce.DecisionTreeEncoder(param_grid={'max_depth': [1, 
    2, 3]}, random_state=2909)), ('discretisation', dsc.
    DecisionTreeDiscretiser(param_grid={'max_depth': [1, 2, 3]},
    random_state=2909, variables=numerical)), ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=100, random_state=0, max_iter=1000))])
house_pipe.fit(X_train, y_train)
X_train_preds = house_pipe.predict(X_train)
X_test_preds = house_pipe.predict(X_test)
print('train mse: {}'.format(mean_squared_error(y_train, X_train_preds,
    squared=True)))
print('train rmse: {}'.format(mean_squared_error(y_train, X_train_preds,
    squared=False)))
print('train r2: {}'.format(r2_score(y_train, X_train_preds)))
print()
print('test mse: {}'.format(mean_squared_error(y_test, X_test_preds,
    squared=True)))
print('test rmse: {}'.format(mean_squared_error(y_test, X_test_preds,
    squared=False)))
print('test r2: {}'.format(r2_score(y_test, X_test_preds)))
