# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
This code is a modified version of the code from this link:
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
His code is a very good one for RNN beginners. Feel free to check it out.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
tf.reset_default_graph()
# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
lstm_train=pd.read_excel("lstm_train.xlsx",header=None,index=False,sheet_name="Sheet1")
lstm_test=pd.read_excel("lstm_test.xlsx",header=None,index=False,sheet_name="Sheet1")


# hyperparameters
lr = 0.003  #学习率
training_iters = 100000  #训练次数
batch_size = 400
n_inputs = 4   # 一行4列
n_steps = 1024    # 输入1024行为一个样本
n_hidden_units = 400   # 隐含层节点数
n_classes = 4      # MNIST classes (0-3 digits)
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None,n_classes])
# Define weights
weights = {
    # (4, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 4)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ]) ),
    # (4, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]) )
}
def RNN(X, weights, biases):
    # hidden layer for input to cell
    ########################################

    # transpose the inputs shape from
    # X ==> (128 batch * 1024 steps, 4 inputs)
    
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 1024 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 1024 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # cell
    ##########################################

    # basic LSTM Cell.
    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm cell is divided into two parts (c_state, h_state)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # You have 2 options for following step.
    # 1: tf.nn.rnn(cell, inputs);
    # 2: tf.nn.dynamic_rnn(cell, inputs).
    # If use option 1, you have to modified the shape of X_in, go and check out this:
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
    # In here, we go for option 2.
    # dynamic_rnn receive Tensor (batch, steps, inputs) or (steps, batch, inputs) as X_in.
    # Make sure the time_major is changed accordingly.
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)

    # hidden layer for output as the final results
    #############################################
    # results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # # or
    # unpack to list [(batch, outputs)..] * steps
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 4)

    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        start=step%(len(lstm_train)-batch_size)
        end=start+batch_size
        batch_xs=lstm_train.iloc[start:end,0:4096]
        batch_ys=lstm_train.iloc[start:end,4096:4100]
        batch_xs=np.array(batch_xs)
        batch_ys=np.array(batch_ys)
        batch_xs=batch_xs.reshape(batch_size,n_steps,n_inputs)
        
        _,loss_,acc_=sess.run([train_op,cost,accuracy], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        start1=step%(len(lstm_test)-batch_size)
        end1=start1+batch_size
        test_xs=lstm_test.iloc[start1:end1,0:4096]
        test_ys=lstm_test.iloc[start1:end1,4096:4100]
        test_xs=np.array(test_xs).reshape(batch_size,n_steps,n_inputs)
        if step % 10 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
            }))
        step += 1
