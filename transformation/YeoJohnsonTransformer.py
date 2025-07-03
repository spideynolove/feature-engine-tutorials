import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feature_engine.imputation import ArbitraryNumberImputer
from feature_engine.transformation import YeoJohnsonTransformer
train_df = pd.read_csv('../data/house-prices/train.csv')
test_df = pd.read_csv('../data/house-prices/test.csv')
X_train = train_df.drop(['Id', 'SalePrice'], axis=1)
y_train = train_df['SalePrice']
X_test = test_df.drop(['Id'], axis=1)
print('X_train :', X_train.shape)
print('X_test :', X_test.shape)
yjt = YeoJohnsonTransformer(variables=['LotArea', 'GrLivArea'])
yjt.fit(X_train)
yjt.lambda_dict_
train_t = yjt.transform(X_train)
test_t = yjt.transform(X_test)
X_train['GrLivArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('GrLivArea')
train_t['GrLivArea'].hist(bins=50)
plt.title('Transformed variable')
plt.xlabel('GrLivArea')
X_train['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')
train_t['LotArea'].hist(bins=50)
plt.title('Variable before transformation')
plt.xlabel('LotArea')
arbitrary_imputer = ArbitraryNumberImputer(arbitrary_number=2)
arbitrary_imputer.fit(X_train)
train_t = arbitrary_imputer.transform(X_train)
test_t = arbitrary_imputer.transform(X_test)
yjt = YeoJohnsonTransformer()
yjt.fit(train_t)
yjt.variables_
yjt.lambda_dict_
train_t = yjt.transform(train_t)
test_t = yjt.transform(test_t)
