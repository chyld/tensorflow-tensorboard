import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

book = xlrd.open_workbook('slr05.xls', encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
N = sheet.nrows - 1

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, name='x')
    y = tf.placeholder(tf.float32, name='y')

with tf.name_scope('variables'):
    m = tf.Variable(0.0, name='m')
    b = tf.Variable(0.0, name='b')

with tf.name_scope('model'):
    y_hat = (m * x) + b
    tf.summary.scalar("y_hat", y_hat)

with tf.name_scope('loss'):
    loss = tf.reduce_sum((y - y_hat) ** 2)
    tf.summary.scalar("loss", loss)
    tf.summary.histogram("loss", loss)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

writer = tf.summary.FileWriter('a04-rpt')
summ = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        for j, k in data:
            sess.run(optimizer, feed_dict={x: j, y: k})

            s = sess.run(summ, feed_dict={x: j, y: k})
            writer.add_summary(s, i)

    m_value, b_value = sess.run([m, b])
    print('m:', m_value, 'b:', b_value)

    writer.add_graph(sess.graph)
