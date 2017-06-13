#!/usr/bin/env python3
# <XGBoost example>
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

# 加载数据
train_df = pd.read_csv('/home/gp/CODES/Kaggle/titanic/train.csv', header=0)
test_df = pd.read_csv('/home/gp/CODES/Kaggle/titanic/test.csv', header=0)

# 填充缺失值(数字型用中值,字符型用众数)
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
  def fit(self, X, y=None):
    self.fill = pd.Series([X[c].value_counts().index[0] 
      if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
      index=X.columns)
    return self
  def transform(self, X, y=None):
    return X.fillna(self.fill)

feature_columns_to_use = ['Pclass', 'Sex', 'Age', 'Fare','Parch']
nonnumeric_columns = ['Sex']
big_X = train_df[feature_columns_to_use].append(test_df[feature_columns_to_use])
big_X_imputed = DataFrameImputer().fit_transform(big_X)

# 将categorical属性数值化
le = LabelEncoder()
for feature in nonnumeric_columns:
  big_X_imputed[feature] = le.fit_transform(big_X_imputed[feature])

# 准备训练,测试集
train_X = big_X_imputed[0:train_df.shape[0]].as_matrix()
test_X = big_X_imputed[train_df.shape[0]::].as_matrix()
train_y = train_df['Survived']

# 训练模型 XGBoost
gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(train_X, train_y)
predictions = gbm.predict(test_X)
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': predictions})
submission.to_csv('/home/gp/CODES/Kaggle/titanic/submit_04.csv', index=False)