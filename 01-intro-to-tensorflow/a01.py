import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
a = tf.constant(4, name="con-a")
b = tf.constant(3, name="con-b")
c = tf.add(a, b, name="add-c")
d = tf.multiply(a, b, name="mul-d")
e = tf.multiply(c, d, name="mul-e")

sess = tf.Session()
res = sess.run(e)
sess.close()

print(res)
