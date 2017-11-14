import nn_load
import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('model/cnn-81.meta')
    new_saver.restore(sess, 'model/cnn-81')
    # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
    pred = tf.get_collection('pred')[0]
    x = tf.get_collection('x')[0]

    graph = tf.get_default_graph()

    #x = graph.get_operation_by_name('x').outputs[0]

    #result = sess.run(y, feed_dict={x: data})
    p = sess.run(pred, feed_dict={x:nn_load.test_data[:128]})
    for data in p:
        if data[0]>=data[1]:
            print(1)
        else:
            print(0)
