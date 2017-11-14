import nn_load
import tensorflow as tf
import numpy as np
def get_train_data(num):
    index_y = np.random.uniform(0,1297,size=num).astype(int)
    index_n = np.random.uniform(0,7103,size=num).astype(int)
    t_y = nn_load.train_y_list[index_y]
    t_n = nn_load.train_n_list[index_n]
    batch_xs = []
    batch_ys = []
    for i in range(num):
        batch_xs.append(t_y[i])
        batch_xs.append(t_n[i])
        batch_ys.append([1,0])
        batch_ys.append([0,1])
    return np.array(batch_xs), np.array(batch_ys)
#batch_xs, batch_ys = get_train_data(64)

image_size = 30
n_h1 = 256
n_h2 = 256
strides = 1
num = 5

# 定义输入输出
with tf.name_scope('input_x'): 
    x = tf.placeholder("float", [None, image_size, image_size, 3], name='x')
with tf.name_scope('label'): 
    y = tf.placeholder("float", [None, 2])

#x_scalar = tf.reshape(x, shape=[-1, image_size*image_size*3])
with tf.name_scope('w_c1'): 
    w_c1 = tf.Variable(tf.random_normal([3, 3, 3, 12]))
with tf.name_scope('b_c1'): 
    b_c1 = tf.Variable(tf.random_normal([12]))

with tf.name_scope('w_c2'): 
    w_c2 = tf.Variable(tf.random_normal([3, 3, 12, 24]))
with tf.name_scope('b_c2'): 
    b_c2 = tf.Variable(tf.random_normal([24]))

with tf.name_scope('w_f1'): 
    w_h1 = tf.Variable(tf.random_normal([7*7*24, n_h1]))
with tf.name_scope('b_f1'): 
    b_h1 = tf.Variable(tf.random_normal([n_h1]))

with tf.name_scope('w_f2'): 
    w_h2 = tf.Variable(tf.random_normal([n_h1, n_h2]))
with tf.name_scope('b_f2'): 
    b_h2 = tf.Variable(tf.random_normal([n_h2]))

with tf.name_scope('w_out'): 
    w_out = tf.Variable(tf.random_normal([n_h2, 2]))
with tf.name_scope('b_out'): 
    b_out = tf.Variable(tf.random_normal([2]))
# 卷积
with tf.name_scope('C1'): 
    conv1 = tf.nn.conv2d(x, w_c1, strides=[1, strides, strides, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b_c1)
    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    out_conv1 = conv1

c1_0 = tf.reshape(out_conv1[:,:,:,0], (2850,14,14,1))
c1_1 = tf.reshape(out_conv1[:,:,:,1], (2850,14,14,1))
c1_2 = tf.reshape(out_conv1[:,:,:,2], (2850,14,14,1))
c1_3 = tf.reshape(out_conv1[:,:,:,3], (2850,14,14,1))
c1_4 = tf.reshape(out_conv1[:,:,:,4], (2850,14,14,1))
c1_5 = tf.reshape(out_conv1[:,:,:,5], (2850,14,14,1))
c1_6 = tf.reshape(out_conv1[:,:,:,6], (2850,14,14,1))
c1_7 = tf.reshape(out_conv1[:,:,:,7], (2850,14,14,1))
c1_8 = tf.reshape(out_conv1[:,:,:,8], (2850,14,14,1))
c1_9 = tf.reshape(out_conv1[:,:,:,9], (2850,14,14,1))
c1_10 = tf.reshape(out_conv1[:,:,:,10], (2850,14,14,1))
c1_11 = tf.reshape(out_conv1[:,:,:,11], (2850,14,14,1))
with tf.name_scope('C2'): 
    conv2 = tf.nn.conv2d(out_conv1, w_c2, strides=[1, strides, strides, 1], padding='SAME')
    conv2 = tf.nn.bias_add(conv2, b_c2)
    conv2 = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    out_conv2 = conv2

c2_0 = tf.reshape(out_conv2[:,:,:,0], (2850,7,7,1))
c2_1 = tf.reshape(out_conv2[:,:,:,1], (2850,7,7,1))
c2_2 = tf.reshape(out_conv2[:,:,:,2], (2850,7,7,1))
c2_3 = tf.reshape(out_conv2[:,:,:,3], (2850,7,7,1))
c2_4 = tf.reshape(out_conv2[:,:,:,4], (2850,7,7,1))
c2_5 = tf.reshape(out_conv2[:,:,:,5], (2850,7,7,1))
c2_6 = tf.reshape(out_conv2[:,:,:,6], (2850,7,7,1))
c2_7 = tf.reshape(out_conv2[:,:,:,7], (2850,7,7,1))
c2_8 = tf.reshape(out_conv2[:,:,:,8], (2850,7,7,1))
c2_9 = tf.reshape(out_conv2[:,:,:,9], (2850,7,7,1))
c2_10 = tf.reshape(out_conv2[:,:,:,10], (2850,7,7,1))
c2_11 = tf.reshape(out_conv2[:,:,:,11], (2850,7,7,1))
c2_12 = tf.reshape(out_conv2[:,:,:,12], (2850,7,7,1))
c2_13 = tf.reshape(out_conv2[:,:,:,13], (2850,7,7,1))
c2_14 = tf.reshape(out_conv2[:,:,:,14], (2850,7,7,1))
c2_15 = tf.reshape(out_conv2[:,:,:,15], (2850,7,7,1))
c2_16 = tf.reshape(out_conv2[:,:,:,16], (2850,7,7,1))
c2_17 = tf.reshape(out_conv2[:,:,:,17], (2850,7,7,1))
c2_18 = tf.reshape(out_conv2[:,:,:,18], (2850,7,7,1))
c2_19 = tf.reshape(out_conv2[:,:,:,19], (2850,7,7,1))
c2_20 = tf.reshape(out_conv2[:,:,:,20], (2850,7,7,1))
c2_21 = tf.reshape(out_conv2[:,:,:,21], (2850,7,7,1))
c2_22 = tf.reshape(out_conv2[:,:,:,22], (2850,7,7,1))
c2_23 = tf.reshape(out_conv2[:,:,:,23], (2850,7,7,1))

with tf.name_scope('reshape'): 
    out_conv = tf.reshape(conv2, [-1, 7*7*24])
# 第一层
with tf.name_scope('F1'): 
    out_h1 = tf.add(tf.matmul(out_conv, w_h1), b_h1)
    out_h1 = tf.nn.relu(out_h1)

# 第二层
with tf.name_scope('F2'): 
    out_h2 = tf.add(tf.matmul(out_h1, w_h2), b_h2)
    out_h2 = tf.nn.relu(out_h2)

with tf.name_scope('out'): 
    out = tf.add(tf.matmul(out_h2, w_out), b_out)
with tf.name_scope('pred'):
    pred = out

with tf.name_scope('loss'): 
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('optimizer'): 
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

with tf.name_scope('accuracy'): 
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)
tf.summary.histogram('w_h1',w_h1)
tf.summary.histogram('b_h1',b_h1)
tf.summary.histogram('w_c1',w_c1)
tf.summary.histogram('b_c1',b_c1)
tf.summary.image('image',x , num)
tf.summary.image('c1_0',c1_0 , num)
tf.summary.image('c1_1',c1_1 , num)
tf.summary.image('c1_2',c1_2 , num)
tf.summary.image('c1_3',c1_3 , num)
tf.summary.image('c1_4',c1_4 , num)
tf.summary.image('c1_5',c1_5 , num)
tf.summary.image('c1_6',c1_6 , num)
tf.summary.image('c1_7',c1_7 , num)
tf.summary.image('c1_8',c1_8 , num)
tf.summary.image('c1_9',c1_9 , num)
tf.summary.image('c1_10',c1_10 , num)
tf.summary.image('c1_11',c1_11, num)

tf.summary.image('c2_0',c2_0, num)
tf.summary.image('c2_1',c2_1, num)
tf.summary.image('c2_2',c2_2, num)
tf.summary.image('c2_3',c2_3, num)
tf.summary.image('c2_4',c2_4, num)
tf.summary.image('c2_5',c2_5, num)
tf.summary.image('c2_6',c2_6, num)
tf.summary.image('c2_7',c2_7, num)
tf.summary.image('c2_8',c2_8, num)
tf.summary.image('c2_9',c2_9, num)
tf.summary.image('c2_10',c2_10, num)
tf.summary.image('c2_11',c2_11, num)
tf.summary.image('c2_12',c2_12, num)
tf.summary.image('c2_13',c2_13, num)
tf.summary.image('c2_14',c2_14, num)
tf.summary.image('c2_15',c2_15, num)
tf.summary.image('c2_16',c2_16, num)
tf.summary.image('c2_17',c2_17, num)
tf.summary.image('c2_18',c2_18, num)
tf.summary.image('c2_19',c2_19, num)
tf.summary.image('c2_20',c2_20, num)
tf.summary.image('c2_21',c2_21, num)
tf.summary.image('c2_22',c2_22, num)
tf.summary.image('c2_23',c2_23, num)

# 定义模型
saver = tf.train.Saver()
tf.add_to_collection('pred', pred)
tf.add_to_collection('x', x)

init = tf.global_variables_initializer()
sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('board/',sess.graph)

sess.run(init)
for epoch in range(2000):
    batch_xs, batch_ys = get_train_data(64)
    sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    if epoch % 5 ==0:
        im,result,ac,c = sess.run([c1_0,merged, accuracy, cost], feed_dict={x: nn_load.test_data, y:nn_load.test_label})
        train_writer.add_summary(result,epoch)
        print("epoch:%s loss:%s accuracy:%s"%(epoch,c,ac))
        saver.save(sess, 'test/cnn', global_step=epoch+1)
        #print(out_conv1)