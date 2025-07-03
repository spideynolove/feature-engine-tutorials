from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from feature_engine.selection import DropHighPSIFeatures
data = pd.read_csv('../data/credit+approval/crx.data', header=None)
data.columns = [('A' + str(s)) for s in range(1, 17)]
data = data.replace('?', np.nan)
data['A2'] = data['A2'].astype('float')
data['A14'] = data['A14'].astype('float')
data['A16'] = data['A16'].map({'+': 1, '-': 0})
data.head()
data['A13'] = data['A13'].map({'g': 'portfolio_1', 's': 'portfolio_2', 'p':
    'portfolio_3'})
data['A13'].fillna('Unknown', inplace=True)
data['A12'] = data['A12'].map({'f': 'wholesale', 't': 'retail'})
data['A12'].fillna('Missing', inplace=True)
data['A6'].fillna('Missing', inplace=True)
labels = {'w': '20-25', 'q': '25-30', 'm': '30-35', 'r': '35-40', 'cc':
    '40-45', 'k': '45-50', 'c': '50-55', 'd': '55-60', 'x': '60-65', 'i':
    '65-70', 'e': '70-75', 'aa': '75-80', 'ff': '85-90', 'j': 'Unknown',
    'Missing': 'Missing'}
data['A6'] = data['A6'].map(labels)
data['date'] = pd.date_range(start='1/1/2018', periods=len(data))
data.head()
vars_cat = data.select_dtypes(include='O').columns.to_list()
vars_cat
for var in vars_cat:
    data[var].value_counts(normalize=True).plot.bar()
    plt.title(var)
    plt.ylabel('% observations')
    plt.show()
vars_num = data.select_dtypes(exclude='O').columns.to_list()
vars_num.remove('A16')
vars_num.remove('date')
vars_num
for var in vars_num:
    data[var].hist(bins=50)
    plt.title(var)
    plt.ylabel('Number observations')
    plt.show()
X_train, X_test, y_train, y_test = train_test_split(data[vars_cat +
    vars_num], data['A16'], test_size=0.1, random_state=42)
transformer = DropHighPSIFeatures(split_frac=0.6, split_col=None, strategy=
    'equal_frequency', threshold=0.1, variables=vars_num, missing_values=
    'ignore')
transformer.fit(X_train)
transformer.cut_off_
transformer.threshold
transformer.psi_values_
transformer.features_to_drop_
tmp = X_train.index <= transformer.cut_off_
sns.ecdfplot(data=X_train, x='A8', hue=tmp)
plt.title('A8 - moderate PSI')
sns.ecdfplot(data=X_train, x='A2', hue=tmp)
plt.title('A2 - low PSI')
X_train.shape, X_test.shape
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
X_train.shape, X_test.shape
X_train, X_test, y_train, y_test = train_test_split(data[vars_cat +
    vars_num], data['A16'], test_size=0.1, random_state=42)
transformer = DropHighPSIFeatures(split_frac=0.5, split_col='A6', strategy=
    'equal_frequency', bins=8, threshold=0.1, variables=None,
    missing_values='ignore')
transformer.fit(X_train)
transformer.variables_
transformer.cut_off_
transformer.psi_values_
transformer.features_to_drop_
tmp = X_train['A6'] <= transformer.cut_off_
sns.ecdfplot(data=X_train, x='A8', hue=tmp)
plt.title('A8 - low PSI')
sns.ecdfplot(data=X_train, x='A15', hue=tmp)
plt.title('A15 - low PSI')
X_train[tmp]['A6'].unique()
X_train[tmp]['A6'].nunique()
len(X_train[tmp]['A6']) / len(X_train)
X_train[~tmp]['A6'].unique()
X_train[~tmp]['A6'].nunique()
len(X_train[~tmp]['A6']) / len(X_train)
X_train.shape, X_test.shape
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
X_train.shape, X_test.shape
X_train, X_test, y_train, y_test = train_test_split(data[vars_cat +
    vars_num], data['A16'], test_size=0.1, random_state=42)
transformer = DropHighPSIFeatures(split_frac=0.5, split_distinct=True,
    split_col='A6', strategy='equal_frequency', bins=5, threshold=0.1,
    missing_values='ignore')
transformer.fit(X_train)
transformer.cut_off_
transformer.psi_values_
transformer.features_to_drop_
tmp = X_train['A6'] <= transformer.cut_off_
sns.ecdfplot(data=X_train, x='A8', hue=tmp)
plt.title('A8 - high PSI')
sns.ecdfplot(data=X_train, x='A15', hue=tmp)
plt.title('A15 - low PSI')
X_train[tmp]['A6'].unique()
X_train[tmp]['A6'].nunique()
len(X_train[tmp]['A6']) / len(X_train)
X_train[~tmp]['A6'].unique()
X_train[~tmp]['A6'].nunique()
len(X_train[~tmp]['A6']) / len(X_train)
X_train.shape, X_test.shape
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
X_train.shape, X_test.shape
X_train, X_test, y_train, y_test = train_test_split(data[vars_cat +
    vars_num], data['A16'], test_size=0.1, random_state=42)
transformer = DropHighPSIFeatures(cut_off=['portfolio_2', 'portfolio_3'],
    split_col='A13', strategy='equal_width', bins=5, threshold=0.1,
    variables=vars_num, missing_values='ignore')
transformer.fit(X_train)
transformer.cut_off_
transformer.psi_values_
transformer.features_to_drop_
tmp = X_train['A13'].isin(transformer.cut_off_)
sns.ecdfplot(data=X_train, x='A3', hue=tmp)
plt.title('A3 - high PSI')
sns.ecdfplot(data=X_train, x='A11', hue=tmp)
plt.title('A11 - high PSI')
sns.ecdfplot(data=X_train, x='A2', hue=tmp)
plt.title('A2 - high PSI')
X_train.shape, X_test.shape
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
X_train.shape, X_test.shape
data['date'].agg(['min', 'max'])
X_train, X_test, y_train, y_test = train_test_split(data[vars_cat +
    vars_num + ['date']], data['A16'], test_size=0.1, random_state=42)
transformer = DropHighPSIFeatures(cut_off=pd.to_datetime('2018-12-14'),
    split_col='date', strategy='equal_frequency', bins=8, threshold=0.1,
    missing_values='ignore')
transformer.fit(X_train)
transformer.cut_off_
transformer.psi_values_
transformer.features_to_drop_
tmp = X_train['date'] <= transformer.cut_off_
sns.ecdfplot(data=X_train, x='A3', hue=tmp)
plt.title('A3 - moderate PSI')
sns.ecdfplot(data=X_train, x='A14', hue=tmp)
X_train.shape, X_test.shape
X_train = transformer.transform(X_train)
X_test = transformer.transform(X_test)
X_train.shape, X_test.shape
