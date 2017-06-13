#!/usr/bin/env python3
#-*- 数据可视化
#-*- heatmap,pairmap,histgram,countplot,boxplot等图像的绘制
#-*- 数据集: titanic_04_train.csv --------- [Pclass, Sex, Age, Fare, Parch]*1309

import numpy as np
import pandas as pd
import titanic.titanic_04 as tt
data = tt.big_X_imputed

##========================= 1.图像绘制练习 =================================##
import matplotlib.pyplot as plt
import seaborn as sns

# sns的context<-{notebook,paper,talk,poster}
# sns的style<-{darkgrid,whitegrid,dark,white,ticks,hls,husl}
# sns的palette<-{deep,muted,bright,pastel,dark,colorblind}
# sns.axes_style() 无参调用会返回当前设置

# 热点图<heatmap>
# 关于相关系数的Heatmap反映了各个属性间的相关性,当相关系数普遍偏小时,说明特征很好.
ax = sns.heatmap(data.astype(float).corr(),
                 vmax=1.0, linewidths=0.1,annot=True,square=True, cmap=plt.cm.viridis)

# 直方图<histgram>
# 可以设置的属性: name,color,virtical, label
ax = sns.distplot(data['Fare'], bins=6)
# 联合分布图<jointplot>
# kind<-{scatter,reg,resid,kde,hex}
ax = sns.jointplot(x='Fare',y='Parch',data=data,kind='kde')
# 种类计数图<countmap>
# 针对categorical属性的统计,若有两个属性就添加hue参数
ax = sns.countplot(x='Pclass',hue='Sex',data=data)

# 箱式图<boxplot>
# x是离散变量,y是连续变量
ax = sns.boxplot(data=data, x='Sex', y='Fare')

# 分组网格<FacetGrid>
# 在一般的两个属性的图像基础上,groupby种类属性,col,row分别代表一个种类属性的groups
# hue的种类属性通过不同颜色表示
g = sns.FacetGrid(data,col='Pclass',row=None,hue='Sex')
g = g.map(plt.scatter, 'Fare','Age')

# 多图网格<subplots>
x = np.linspace(0, 2*np.pi, 400)
y = np.sin(x**2)
fig,axes = plt.subplots(2,2, subplot_kw=dict(polar=True))   # 构造2*2=4个子图
axes[0,0].plot(x,y)
axes[1,1].scatter(x,y)
