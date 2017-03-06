# TensorFlow的学习记录

刚一读研，TF发布正好半年，所以我有幸能直接通过这一神器学习DeepLearning。除了极客学院的官方文档中文翻译版以外，我在B站上搜到了一个很棒的TF入门视频教程——《TFgirls》。

* 视频链接：<http://www.bilibili.com/video/av6642102/>
* 作者github链接： <https://github.com/CreatCodeBuild/TensorFlow-and-DeepLearning-Tutorial>

该视频并没有详细地讲解DeepLearning的算法知识，但很简洁地介绍了TF的基本用法，对初次接触TF的小伙伴能起到很好的引领作用。
以下是我学过后的一些总结：

---

## TF简介

TensorFlow的运作原理是一种静态学习机制，由两部分构成：计算图谱(Graph)，会话(Session)。

首先，要在Graph中架构整个计算流程，然后将Graph交给Session运行。底层中，Seesion是将Graph中定义的运算交予GPU并行处理，最后将结果统一返回给程序。这样可以充分发挥GPU的运算能力，并尽可能减少数据传输造成的时间浪费。

静态学习的效率要高于动态学习（减少了数据传输过程），但Graph一但确定，无法根据实际数据集结构动态变化。所以通常情况下模型和数据集是不变统一的。

## 构造计算图谱
TF的一大重点就是Graph的构造，在这步要定义整个学习模型的计算流程，主要有以下节点：
* placeholder ： 数据集的节点，包括训练集样本标签和测试集样本标签;
* model ：模型函数，返回计算结果logit；
* constant ：模型的常量；
* Variable ：模型的参数，训练过程中要优化的部分；
* loss ：损失函数（logit，labels）;
* optimizer : 优化算法，修改Variable，对loss优化;

## 数据集切割
对于大量数据样本，要切割数据集成多个批次(batch)，在Session中每次for循环传入一组batch。TFgirls中使用生成器将数据集分批：

```
#! /usr/bin/env python3
'''
分批处理大量数据的输入
关键点：placeholder，生成器,batch
'''
import tensorflow as tf

def create_big_data():
    return [-x for x in range(1000)]

#通过生成器将大数据分批取出
def load_batch(data,step):
    index = 0
    while index < len(data):
        yield data[index:index+step]
        index += step
    return

graph = tf.Graph()
with graph.as_default():
    value1 = tf.placeholder(dtype=tf.float64)
    value2 = tf.Variable([3,4],dtype=tf.float64)
    mul = value1 * value2

with tf.Session(graph=graph) as mySess:
    init = tf.global_variables_initializer()
    mySess.run(init)
    data = create_big_data()
    for batch in load_batch(data,2):
        result = mySess.run(mul,feed_dict={value1:batch})
        print('乘法（value1,value2）= ',result)
    
```

---

## TF可视化
Tensorboard是一个非常强大的可视化工具。它将计算图谱和训练中参数的变化过程记录为events文件，并可以通过浏览器查看。
* **在初始化结尾创建一个FileWriter**
```
writer = tf.summary.FileWriter('./board',graph=graph)
```
*注意，这行代码要放在graph创建之后。*
* **添加名称封装图谱节点**
```
with tf.name_scope('INPUT'):
```
对于一些重要的节点添加名称并模块化，可以更有条理的分析计算图谱。
* **为变量添加summary**
```
tf.scalar_summary("loss",loss)                  #标量
tf.histogram_summary('fc_weights',fc_weights)   #直方图
```
在定义图谱的结尾，通过定义merge-summary运算，将个变量的summary聚合到一起。
```
merged = tf.merge_all_summaries()
```
因为merge-summary是运算，所以要到 Session 中运行。最后在将运算结果添加到FileWriter里。
```
summary = mySess.run(merged)
writer.add_summary(summary,i)   #此步要在for循环中多次执行
```
* **shell中可以打开events的链接**
```
gp@gp-pc:~$ tensorboard --logdir board
```
*注意：merge-summary是一个运算，有定义，要运行，还要写道board记录文件中。*

---


## 全连接模型（FullyConnected）
全连接模型属于线性模型的一种：$y=w*x+b$,其中x，y是数据集和标签集，W，b是模型参数。

tf中的构造过程如下：

1. **定义一个权重矩阵weights**

```
fc_weights = tf.Variable(tf.truncated_normal([shape[1] * shape[2] * shape[3],hidden],stddev=0.1))
```
这里 `shape[1] * shape[2] * shape[3]` 表示将一个样本数据展开成一维，`hidden` 是隐藏层的大小，通常将W初始化成正则分布。

2. **定义一个偏差向量biases**

```
fc_biases = tf.Variable(tf.constant(0.1,shape=[hidden]))
```
biases是一维向量，长度和 fc_weights 的 hidden 一致.若该层是神经网络的最后一层，hidden 的取值应和 labels 的维度一致。
例如，labels 是 one-hot 数据，维度是10,则 hidden=10.

3. **定义模型**

```
fc_model = tf.matmul(samples, fc_weights) + fc_biases
```
使用matmul进行矩阵乘法运算。注意这里的 samples 是展开 shape[1,2,3] 后的数据 batch.