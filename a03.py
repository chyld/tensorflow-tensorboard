import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

a = tf.constant(3)
b = tf.constant(4)
c = a * b
d = c + a

with tf.Session() as sess:
    r = sess.run(d)
    print('r:', r)
    print('a:', a, 'b:', b, 'c:', c, 'd:', d)
