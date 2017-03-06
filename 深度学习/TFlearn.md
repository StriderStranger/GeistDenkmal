# TensorFlow的学习记录

刚一读研，TF发布正好半年，所以我有幸能直接通过这一神器学习DeepLearning。除了极客学院的官方文档中文翻译版以外，我在B站上搜到了一个很棒的TF入门视频教程——《TFgirls》。

* 视频链接：<http://www.bilibili.com/video/av6642102/>
* 作者github链接： <https://github.com/CreatCodeBuild/TensorFlow-and-DeepLearning-Tutorial>

该视频并没有详细地讲解DeepLearning的算法知识，但很简洁地介绍了TF的基本用法，对初次接触TF的小伙伴能起到很好的引领作用。
以下是我学过后的一些总结：

---

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

