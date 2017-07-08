import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

X_orig = np.random.randn(2000, 3)
w_orig = np.array([[0.3, 0.5, 0.1]])
b_orig = -0.2
e_orig = np.random.randn(2000, 1) * 0.1
y_orig = (X_orig @ w_orig.T) + b_orig + e_orig

with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=[None, 3], name='X')
    y = tf.placeholder(tf.float32, shape=[None, 1], name='y')

with tf.name_scope('variables'):
    weights = tf.Variable([[0, 0, 0]], dtype=tf.float32, name='weights')
    bias = tf.Variable(0, dtype=tf.float32, name='bias')

with tf.name_scope('prediction'):
    y_hat = (weights @ tf.transpose(X)) + bias

with tf.name_scope('loss'):
    loss = tf.reduce_mean((y - y_hat) ** 2)
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)

writer = tf.summary.FileWriter('a05-rpt')
summ = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(optimizer, feed_dict={X: X_orig, y: y_orig})
        s = sess.run(summ, feed_dict={X: X_orig, y: y_orig})
        writer.add_summary(s, i)

    writer.add_graph(sess.graph)
    w_final, b_final = sess.run([weights, bias])
    print('weights:', w_final, 'bias:', b_final)
