#coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import csv

def sub_list(list1,list2):
    num1 = len(list1)
    num2 = len(list2)
    list3 = []
    if num1 == num2:
        for i in range(num1):
            list3.append((list1[i]) - int(list2[i]))
        return list3
    else:
        print('列表1长度 =', num1)
        print('列表2长度 =', num2)
        print('列表长度不相同，无法相加')
        return -1

#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=90):
    batch_index=[]
    data_train=data[train_begin:train_end]
    #normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    normalized_train_data=data_train
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:input_size]
       y=normalized_train_data[i:i+time_step,input_size:]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

#获取测试集
def get_test_data(time_step=20,test_begin=5800):
    print('time_step is ', time_step)
    print('train_end is ', train_end)
    data_test=data[train_end:]
    normalized_test_data = data_test
    print('normalized_test_data len is',len(normalized_test_data))
    #mean=np.mean(data_test,axis=0)
    #std=np.std(data_test,axis=0)
    #normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    print('size is ', size)
    test_x,test_y=[],[]
    for i in range(size-1):
        x=normalized_test_data[i*time_step:(i+1)*time_step,:input_size]
        y=normalized_test_data[i*time_step:(i+1)*time_step,input_size:]
        test_x.append(x.tolist())
        test_y.extend(y.tolist())
    test_x.append((normalized_test_data[(i+1)*time_step:,:input_size:]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,input_size:]).tolist())

    '''
    print('test_x len is',len(test_x))
    print('test_x[7] len is',len(test_x[7]))
    print('test_x[0][0] len is', len(test_x[0][0]))
    print('test_y len is',len(test_y))
    print('test_y[0] len is', len(test_y[0]))
    print('len(test_y)/output_size is ',len(test_y)/output_size)
    print(test_y)
    exit()
    '''
    #return mean,std,test_x,test_y
    return test_x, test_y

# ——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size = tf.shape(X)[0]
    time_step = tf.shape(X)[1]
    w_in = weights['in']
    b_in = biases['in']
    input = tf.reshape(X, [-1, input_size])  # 需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, rnn_unit])  # 将tensor转成3维，作为lstm cell的输入
    cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, rnn_unit])
    w_out = weights['out']
    b_out = biases['out']
    pred = tf.matmul(output, w_out) + b_out
    return pred, final_states

# ————————————————训练模型————————————————————

def train_lstm(batch_size=60, time_step=20, train_begin = 0, train_end = 5000):
    X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, output_size])
    batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)

    with tf.variable_scope("sec_lstm"):
        pred, _ = lstm(X)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))
    train_op = tf.train.AdamOptimizer(lr).minimize(loss,global_step=global_step)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter):  # 这个迭代次数，可以更改，越大预测效果会更好，但需要更长时间
            for step in range(len(batch_index) - 1):
                _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x[batch_index[step]:batch_index[step + 1]],
                                                                 Y: train_y[batch_index[step]:batch_index[step + 1]]})
            if i % 10 == 0:
                print('global_step:',sess.run(global_step), ' lr:', sess.run(lr))
                print("Number of iterations:", i, " loss:", loss_)
            if i % 2000 == 0 or i == iter - 1:
                print("model_save: ", saver.save(sess, 'model_save2/modle.ckpt'))
    print("The train has finished")

#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #mean,std,test_x,test_y=get_test_data(time_step)
    test_x, test_y = get_test_data(time_step)
    #with tf.variable_scope("sec_lstm",reuse=True):    #在训练和预测一起运行时使用
    with tf.variable_scope("sec_lstm"):               #在单独运行预测时使用
        pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model_save2')
        saver.restore(sess, module_file)
        test_predict=[]
        print(len(test_x)-1)
        for step in range(len(test_x)-1):
            '''
            if step == len(test_x)-2:
                for i in range(len(test_x[step])):
                    test_x[step][i] = sub_list(test_x[step][i],test_x[step][i])
            '''
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})
            predict=prob.reshape((-1))
            test_predict.extend(predict)
    #test_y=np.array(test_y)*std[7]+mean[7]
    #test_predict=np.array(test_predict)*std[7]+mean[7]
    #acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差程度
    print(test_y)

    test_y = list(itertools.chain(*test_y))
    acc = np.average(np.abs(sub_list(test_predict,test_y[:len(test_predict)])))  # 偏差程度
    print("The accuracy of this predict:",acc)
    for i in range(len(test_predict)):
        test_predict[i] = int(test_predict[i])
    return test_y[:len(test_predict)], test_predict

    '''
    #以折线图表示结果
    plt.figure()
    plt.plot(list(range(len(test_predict))), test_predict, color='b',)
    plt.plot(list(range(len(test_y))), test_y,  color='r')
    plt.show()
    '''

if __name__ == "__main__":
    rnn_unit=10       #隐层数量
    #lr=0.06         #学习率
    global_step = tf.Variable(0, trainable=False)
    lr =tf.train.exponential_decay(0.9, global_step, decay_steps=100, decay_rate=0.995, staircase=True)
    iter = 40000
    batch_size = 10
    time_step = 5
    train_begin = 0
    train_end = 90
    #——————————————————导入数据——————————————————————
    f=open('data.csv')
    df=pd.read_csv(f)     #读入股票数据
    data=df.iloc[:,1:].values  #取第3-10列

    input_size = int(data.shape[1]/3*2)
    output_size = int(data.shape[1]/3)
    data_len = data.shape[0]
    # ——————————————————打印——————————————————————
    print('rnn_unit is',rnn_unit)
    print('iter is', iter)
    print('batch_size is', batch_size)
    print('time_step is', time_step)
    print('train_begin is', train_begin)
    print('train_end is',train_end)
    print('input_size is', input_size)
    print('output_size is', output_size)
    print('data_size is', data.shape[1])

    #——————————————————定义神经网络变量——————————————————
    #输入层、输出层权重、偏置

    weights={
             'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
             'out':tf.Variable(tf.random_normal([rnn_unit,output_size]))
            }
    biases={
            'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
            'out':tf.Variable(tf.constant(0.1,shape=[output_size,]))
           }

    train_lstm( batch_size, time_step, train_begin, train_end)
    test_y, test_predict = prediction(time_step)


    print('len(test_predict) is ',len(test_predict))
    predict = []
    for i in range(int(len(test_predict)/output_size)):
        predict.append(list(test_predict[i*output_size:(i+1)*output_size+1]))
    print(len(predict))
    print(len(predict[0]))
    for i in range(int(len(test_predict)/output_size)):
        print(sum(predict[i]))
    #'''
    path = r'E:\1a_lyy\RPS资源计划\新建文件夹\rnn程序\分析数据\RPS_VS2_机组计划量_工作时间\predict.csv'
    file = []
    for i in range(int(len(test_y)/output_size)):
        file.append('第' + str(i+1) + '组数据')
        file.append(test_y[i*output_size:(i+1)*output_size+1])
        file.append(test_predict[i * output_size:(i + 1) * output_size + 1])
    with open(path, 'w+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in file:
            writer.writerow(row)
    #'''
