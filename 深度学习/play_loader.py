#!/usr/bin/env python3
'''
体验tf.Saver的用法
[一些总结]
* ckpt在理解上可以看作“变量名”和对应的“值”的字典
* restore时不需要对变量初始化
* 变量名称很重要，ckpt文件的变量和待载入的变量的name必须一致
* 可以给Saver初始化函数传递字典表达“变量名”和“新变量”的对应关系
>> saver = tf.train.Saver({'var_name':weights})
@iris >_<
'''

import tensorflow as tf
import numpy as np

############################### SAVE #######################################
'''
# 创建一些变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='weight')
biases = tf.Variable(tf.zeros([200]), name='biases')

# 添加一个初始化op
init = tf.global_variables_initializer()

# 给所有变量添加Saver op
saver = tf.train.Saver()

# Later, 保存变量到磁盘
with tf.Session() as sess:
    sess.run(init)
    save_path = saver.save(sess, 'model_variable.ckpt')
    print('Model saved in file: ',save_path)

# 此时生成了4个文件： 
# checkpoint, model_variable.ckpt.data-00000-of-00001,
# model_variable.ckpt.index, model_variable.ckpt.meta
'''
############################### RESTORE #######################################
'''
# 创建一些变量
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='v1')
biases = tf.Variable(tf.zeros([200]), name='v2')

# 恢复变量时不需要对它们做初始化

# 给所有变量添加Saver op
saver = tf.train.Saver({'weight':weights, 'biases':biases})

# Later, 从磁盘中加载变量
with tf.Session() as sess:
    saver.restore(sess, 'model_variable.ckpt')
    print('Model restored.')
    print('weights: ')
    print(sess.run(weights))
'''
############################### 查看 #######################################
'''
# 查看ckpt文件中的变量名和对应值
from tensorflow.python import pywrap_tensorflow
checkpoint_path = 'model_variable.ckpt'
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print('tensor_name: ',key)
    print(reader.get_tensor(key))
'''