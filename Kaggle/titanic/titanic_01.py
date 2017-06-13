#!/usr/bin/env python3
# <Titanic Data Science Solutions>

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

train_df = pd.read_csv('/home/gp/CODES/Kaggle/titanic/train.csv')
test_df = pd.read_csv('/home/gp/CODES/Kaggle/titanic/test.csv')

##====================================================================##

# 1. 删去无关属性:**ticket**,**cabin**
train_df = train_df.drop(['Ticket','Cabin'], axis=1)
test_df = test_df.drop(['Ticket','Cabin'], axis=1)
combine = [train_df, test_df]

# 2. 从**Name**中提取**Title**信息并数值化
for dataset in combine:
  dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in combine:
  dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
  dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
  dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	  'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for dataset in combine:
  dataset['Title'] = dataset['Title'].map(title_mapping)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df,test_df]

# 3. 数值化**sex**
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

# 4. 补完**age**的缺失值 (根据**sex**和**pclass**预测缺失值)
guess_ages = np.zeros((2,3))
for dataset in combine:
  for i in range(0,2):
    for j in range(0,3):
      guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass']==j+1)]['Age'].dropna()  # 去掉nan数据
      age_guess = guess_df.median()
      guess_ages[i,j] = int(age_guess/0.5 + 0.5) * 0.5        # 四舍五入到0.5精度
  for i in range(0,2):
    for j in range(0,3):
      dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex==i) & (dataset.Pclass==j+1), 'Age' ] = guess_ages[i,j]
  dataset['Age'] = dataset['Age'].astype(int)

  # **age**离散化
train_df['AgeBand'] = pd.cut(train_df['Age'],5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()
for dataset in combine:
  dataset.loc[ dataset['Age']<=16, 'Age' ] = 0
  dataset.loc[ (dataset['Age']>16) & (dataset['Age']<=32), 'Age' ] = 1
  dataset.loc[ (dataset['Age']>32) & (dataset['Age']<=48), 'Age' ] = 2
  dataset.loc[ (dataset['Age']>48) & (dataset['Age']<=64), 'Age' ] = 3
  dataset.loc[ dataset['Age']>64, 'Age' ] = 4
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# 5. 将**SibSp**和**Parch**合并成**FamilySize**
for dataset in combine:
  dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in combine:
  dataset['IsAlone'] = 0
  dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
train_df = train_df.drop(['Parch','SibSp','FamilySize'], axis=1)
test_df = test_df.drop(['Parch','SibSp','FamilySize'], axis=1)
combine = [train_df, test_df]

# 6. 将**age**和**pclass**合并成**age*pclass**
for dataset in combine:
  dataset['Age*Class'] = dataset.Age * dataset.Pclass

# 7. 补完**embarked**的缺失值(用众数); 并数值化
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
  dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
for dataset in combine:
  dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

# 8. 补完**fare**的缺失值;并离散化
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean()
for dataset in combine:
  dataset.loc[ dataset['Fare']<=7.91, 'Fare' ] = 0
  dataset.loc[ (dataset['Fare']>7.91) & (dataset['Fare']<=14.454), 'Fare' ] = 1
  dataset.loc[ (dataset['Fare']>14.454) & (dataset['Fare']<=31), 'Fare' ] = 2
  dataset.loc[ dataset['Fare']>31, 'Fare' ] = 3
  dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

# 构建样本标签
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()

##=======================================================================##

# LR模型
logreg = LogisticRegression()
logreg.fit(X_train,Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print("accuracy of LR:",acc_log)

# 查看相关性
coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation',ascending=False)

# SVM模型
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train,Y_train) * 100, 2)
print("accuracy of SVM:",acc_svc)

# GNB模型
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100,2)
print("accuracy of GNB:",acc_gaussian)

# Perceptron模型
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train,Y_train) * 100, 2)
print("accuracy of Percep:",acc_perceptron)

# Linear SVC模型
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train,Y_train) * 100, 2)
print("accuracy of LSVC:",acc_linear_svc)

# SGD模型
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print("accuracy of SGD:",acc_sgd)

# DT模型
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_dtree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print("accuracy of DTree:",acc_dtree)

# RF模型
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_RF = round(random_forest.score(X_train, Y_train) * 100, 2)
print("accuracy of RF:",acc_RF)

# 模型评估
models = pd.DataFrame({
  'Model': ['SVM', 'LR', 'RF', 'GNB', 'Percep', 'SGD', 'LSVC', 'DT'],
  'Score': [acc_svc,acc_log,acc_RF,acc_gaussian,acc_perceptron,acc_sgd,acc_linear_svc,acc_dtree]})
models.sort_values(by='Score', ascending=False)

# 选择acc最大的作为submission
submission = pd.DataFrame({
  'PassengerId': test_df["PassengerId"],
  'Survived': Y_pred })

submission.to_csv('/home/gp/CODES/Kaggle/titanic/submit_01.csv',index=False)