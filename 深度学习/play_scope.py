#!/usr/bin/env python3
'''
区分name.scope和varialbe.scope
[一些总结]
* name_scope对get_variable()创建的变量name没有影响
* name_scope和variable_scope都会给op的name加上前缀 
* variable_scope都会添加前缀，会返回一个op
@iris >_<
'''

import tensorflow as tf
import numpy as np
slim = tf.contrib.slim

'''
with tf.name_scope('hello') as name_scope:
    arr1 = tf.get_variable(shape=[2,10], dtype=tf.float32, name='arr1')
    print(name_scope)
    print(arr1.name)
    print('scope_name:%s ' % tf.get_variable_scope().original_name_scope)
'''
'''
with tf.variable_scope('world') as variable_scope:
    arr1 = tf.get_variable(shape=[2,10], dtype=tf.float32, name='arr1')
    print(variable_scope)
    print(variable_scope.name)
    print(arr1.name)
    print('scope_name:%s ' % tf.get_variable_scope().original_name_scope)
    with tf.variable_scope('xixi') as v_scope2:
        print('scope_name:%s ' % tf.get_variable_scope().original_name_scope)
'''
'''
with tf.name_scope('name1'):
    with tf.variable_scope('var1'):
        w = tf.get_variable(shape=[2], name='w')
        res = tf.add(w,[3])
print(w.name)
print(res.name)
'''

'''
with tf.name_scope('name1'):
    with tf.variable_scope('var1'):
        img = tf.placeholder(tf.float32, [None,300,300,1], name='placeholder')
        layer = slim.conv2d(img, 3, [3,3], scope='conv6')
print(img.name)
print(layer.name)
'''