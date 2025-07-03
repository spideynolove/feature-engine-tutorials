import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from feature_engine.transformation import LogCpTransformer
X, y = fetch_california_housing(return_X_y=True)
X = pd.DataFrame(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
    random_state=0)
print('Column names:', list(X_train.columns))
print("""
Column positions:""")
for i, col in enumerate(X_train.columns):
    print(f'{i}: {col}')
num_feats = [6, 7]
tf = LogCpTransformer(variables=num_feats, C='auto')
tf.fit(X_train)
train_t = tf.transform(X_train)
test_t = tf.transform(X_test)
plt.figure(figsize=(12, 12))
for idx, col in enumerate(num_feats, start=1):
    plt.subplot(2, 2, round(idx * 1.4))
    plt.title(f'Untransformed variable {col}')
    X_train[col].hist()
    plt.subplot(2, 2, idx * 2)
    plt.title(f'Transformed variable {col}')
    train_t[col].hist()
tf.variables_
tf.C_
