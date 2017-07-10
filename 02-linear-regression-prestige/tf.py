import statsmodels.api as sm
prestige = sm.datasets.get_rdataset("Duncan", "car", cache=True).data

X1 = prestige[['income', 'education']].astype(float).as_matrix()
# X = sm.add_constant(X)
y1 = prestige['prestige'].as_matrix()

from util import *

# because X will be shuffled, y has to be attached
y2 = y1.reshape(45,1)
matrix = np.append(X1, y2, axis=1)

### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###
### ------------------------------------------------------------- ###

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def execute(learning_rate):
    tf.reset_default_graph()

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
        y = tf.placeholder(tf.float32, shape=[1, None], name='y')

    with tf.name_scope('variables'):
        weights = tf.Variable([[0, 0]], dtype=tf.float32, name='weights')

    with tf.name_scope('prediction'):
        y_hat = weights @ tf.transpose(X)
        # tf.summary.histogram('y_hat', y_hat)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean((y - y_hat) ** 2)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    writer = tf.summary.FileWriter('prestige-rpt/lr' + str(learning_rate))
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer.add_graph(sess.graph)
        for i in range(10):
            for rows in shuffle_batch(matrix, 20):
                XX = rows[:,[0,1]]
                yy = np.asmatrix(rows[:, 2])
                sess.run(optimizer, feed_dict={X: XX, y: yy})
                s = sess.run(summary, feed_dict={X: XX, y: yy})
                writer.add_summary(s, i)

            print('--->', i, sess.run([weights]))

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
for lr in learning_rates:
    execute(lr)

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
