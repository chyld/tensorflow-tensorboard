import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_orig = np.random.randn(2000, 3)
w_orig = np.array([[0.3, 0.5, 0.1]])
b_orig = -0.2
e_orig = np.random.randn(1, 2000) * 0.1
y_orig = (w_orig @ X_orig.T) + b_orig + e_orig

def execute(learning_rate):
    tf.reset_default_graph()

    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
        y = tf.placeholder(tf.float32, shape=[1, None], name='y')

    with tf.name_scope('variables'):
        weights = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
        bias = tf.Variable(0, dtype=tf.float32, name='bias')
        tf.summary.scalar('bias', bias)

    with tf.name_scope('prediction'):
        y_hat = (weights @ tf.transpose(X)) + bias
        tf.summary.histogram('y_hat', y_hat)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean((y - y_hat) ** 2)
        tf.summary.scalar('loss', loss)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    writer = tf.summary.FileWriter('a05-rpt/lr' + str(learning_rate))
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        writer.add_graph(sess.graph)
        for i in range(10):
            sess.run(optimizer, feed_dict={X: X_orig, y: y_orig})
            s = sess.run(summary, feed_dict={X: X_orig, y: y_orig})
            writer.add_summary(s, i)

            if i == 9:
                print('--->', i, sess.run([weights, bias]))

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###

learning_rates = [0.5, 0.05, 0.005, 0.0005]
for lr in learning_rates:
    execute(lr)

### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
### ------------------------------------------------------------------------------------------- ###
