#!/usr/bin/env python3
'''
机器学习基本框架
没有数据，并不能真正训练
'''
import tensorflow as tf
import numpy as np
# import load
# TestData = load.TestData

# def generate(TestData,step):
#     index = 0
#     i=1
#     while index < len(TestData):
#         yield i,TestData.X[index:index+step], TestData.Y[index:index+step]
#         index = index + step
#         i = i + 1
#     return

def accuracy(logits,labels):
    return np.sum(logits==labels) / labels.shape[0]

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('SAMPLES'):
        samples = tf.placeholder('float', shape=[None,784])
    with tf.name_scope('LABELS'):
        labels  = tf.placeholder('float', shape=[None,10])
    with tf.name_scope('FC_MODEL'):
        with tf.name_scope('FC_W'):
            fc_weights = tf.Variable(tf.truncated_normal([784,10], stddev=1.0))
        with tf.name_scope('FC_B'):
            fc_biases = tf.Variable(tf.constant(0.1,shape=[10]))
        with tf.name_scope('LOGITS'):
            logits = tf.nn.relu(tf.matmul(samples,fc_weights) + fc_biases)
    with tf.name_scope('LOSS'):
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits,labels) )
    with tf.name_scope('OPTIMIZER'):
        optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

writer = tf.summary.FileWriter('./board',graph=graph)

with tf.Session(graph=graph) as mySess:
    mySess.run(tf.global_variables_initializer())
    # for i,sample_batch,label_batch in generate(TestData,100):
    #     opt,lgt = sess.run([optimizer,logits],feed_dict={samples:sample_batch,labels:label_batch})
    #     accuracy = accuracy(lgt, labels)     #每一批的accuracy
    #     print('accuracy = %g'%(accuracy))

print('Training Successful !!')
