#!/usr/bin/env python3
'''
体验tf.pad的用法
[一些总结]
* pad函数用来给Tensor镶边，主要由paddings参数控制
* paddings是一个序偶数组，待处理的Tensor有dim维，则paddings就有dim个序偶
* 每个序偶[a,b]：a表示在维度前添加a组0， b表示在维度后添加b组0
@iris >_<
'''

import tensorflow as tf
import numpy as np

npimg = np.random.randint(1,20,[5,5])
img = tf.constant(npimg)
paddings = [[2,0], [1,1]]
img = tf.pad(img, paddings, mode='CONSTANT')

with tf.Session() as sess:
    img = sess.run(img)
    print(img)
    print(img.shape)


npimg2 = np.random.randint(1,20,[3,3,3])
img2 = tf.constant(npimg2)
paddings = [[1,1], [1,1], [1,1]]
img2 = tf.pad(img2, paddings,mode='CONSTANT')

with tf.Session() as sess:
    img2 = sess.run(img2)
    print(img2)
    print(img2.shape)
