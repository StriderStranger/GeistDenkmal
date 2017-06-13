# Titanic Data Science Solutions
## 项目描述
[https://www.kaggle.com/startupsci/titanic-data-science-solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions)
## 目标
对于测试集中的每个 PassengerID,预测对应 Survived 属性的值(0或1).
## 评价指标
(accuracy): 预测正确的乘客数的百分比.
## 数据集
* train.csv
* test.csv

(variable): passengerID, survival, pclass, sex, age, sibsp, parch, ticket, fare, cabin, embarked 
## 性质分析
* Categorical: **survived**(0,1), **sex**(male,female), **embarked**(S,C,Q)
* Ordinal: **pclass**(1,2,3)
* Continuous: **age**(0.4 : 80), **fare**(0 : 521.3)
* Discrete: **sibsp, **parch**
* Mixed: **ticket, **cabin**
## 错误异常分析
* errors: **name**
* empty: **age**, **cabin**, **embarked** in train.csv;  **age**, **cabin**, **fare** in test.csv
## 初步结论
1. 补全**age**,**embarked**；抛弃**cabin**因为不完全程度太高.
2. 抛弃**ticket**,**passengerId**,**name**因为对分类贡献不大.
3. 离散化**age**和**fare**.
4. 构造**family**根据**parch**和**sibsp**.
5. 女性(sex=female),儿童(age<?),上层人士(pclass=1) 更可能会获救(survived=1).

## 相关性分析
判断离散值属性(少量取值)和标签(0,1)的相关性
```python
# 判断Pclass(1,2,3)和Survived(0,1)的相关性
train_df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean()
# 判断Sex和Survived的相关性
train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean()
# 判断SibSp和Survived的相关性
train_df[['SibSp','Survived']].groupby(['SibSp'],as_index=False).mean()
# 判断Parch和Survived的相关性
train_df[['Parch','Survived']].groupby(['Parch'],as_index=False).mean()
```
## 可视化分析
判断连续值属性与标签(0,1)的相关性
```python
# age直方图 groupby(survived=0, survived=1)
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
# age直方图 groupby(pclass & survived)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# pclass-survived散点图 groupby(embarked & sex)
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# fare条形图 groupby(embarked & survived & sex)
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
```

## Wrangle Data
1. 删去无关属性:**ticket**,**cabin**
2. 从**Name**中提取**Title**信息并数值化
```python
pd.crosstab(train_df['Title'], train_df['Sex'])     # 做title和sex的交叉表,查看title分布
train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean()
```
3. 数值化**sex**
4. 补完**age**的缺失值 (根据**sex**和**pclass**预测缺失值);并离散化
```python
# 查看age与sex和pclass的相关性
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# 查看age的bands
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
```
5. 将**SibSp**和**Parch**合并成**FamilySize**
```python
train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False),mean()
```
6. 将**age**和**pclass**合并成**age*pclass**
7. 补完**embarked**的缺失值(用众数); 并数值化
8. 补完**fare**的缺失值;并离散化
```python
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
```
9. 用LR查看属性和类别相关性
```python
logreg.fit(X_train, Y_train)
coeff_df = pd.DataFrame(X_train.columns)
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
```
10. 应用各种算法模型,并进行评估

# 样本结构
(训练集):
|Survived|Pclass|Sex|Age|Fare|Embarked|Title|IsAlone|Age*Class|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|[0,1]|[1,2,3]|[0,1]|[0..4]|[0..3]|[0,1,2]|[1..5]|[0,1]|[0..4]x[1..3]

(测试集): 
|PassengerId|Pclass|Sex|Age|Fare|Embarked|Title|IsAlone|Age*Class|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|[892..1309]|[1,2,3]|[0,1]|[0..4]|[0..3]|[0,1,2]|[1..5]|[0,1]|[0..4]x[1..3]

# 模型选择
* LR: 逻辑回归
* KNN: k最临近算法
* SVM: 支持向量机
* NB: 朴素贝叶斯分类
* DT: 决策树
* RF: 随机森林
* Perceptron: 线性感知器
* ANN: 人工神经网络
* RVM: 相关向量机