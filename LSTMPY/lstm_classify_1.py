import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import tensorflow as tf
 
#定义常量
rnn_unit=10       #hidden layer units
input_size=4
output_size=1
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
#f=open('lstm_train.xlsx')
data=pd.read_excel("lstm_train_1.xlsx",sheet_name="Sheet1",header=None)

batch_size=80 #每一次训练80行
time_step=20  #每256行作为一个序列，加上标签 ,一个样本1024*4行分为4行，每行256*4个数据


 
 
#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=5800):
    batch_index=[]
   # data_train=data[train_begin:train_end]
   # normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]   #训练集
    for i in range(len(data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=data.iloc[i:i+time_step,0:data.shape[1]-1]
       y=data.iloc[i:i+time_step,data.shape[1]-1:data.shape[1]]
       train_x.append(x)
       train_y.append(y)
    batch_index.append((len(data)-time_step))
    return batch_index,train_x,train_y
 
 
 
#获取测试集
def get_test_data(time_step=20,test_begin=5800):
    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample
    test_x,test_y=[],[] 
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:7]
       y=normalized_test_data[i*time_step:(i+1)*time_step,7]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:7]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,7]).tolist())
    return mean,std,test_x,test_y
 
 
 
#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置
 
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }
 
#——————————————————定义神经网络变量——————————————————
def lstm(X):    
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in'] 
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states
 
 
 
#——————————————————训练模型——————————————————
def train_lstm(batch_size=80,time_step=32,train_begin=2000,train_end=5800):
    tf.reset_default_graph()
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    # 训练样本中第2001 - 5785个样本，每次取15个
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
   # print(np.array(train_x).shape)# 3785  15  7
    #print(batch_index)
    #相当于总共3785句话，每句话15个字，每个字7个特征（embadding）,对于这些样本每次训练80句话
    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15) 
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #重复训练200次
        for i in range(2000):
            #每次进行训练的时候，每个batch训练batch_size个样本
            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={
                        X:np.array(train_x[batch_index[step]:batch_index[step+1]]),
                        Y:np.array(train_y[batch_index[step]:batch_index[step+1]])
                        })
            print(i,loss_)
            if i % 2000==0:
                print("保存模型：",saver.save(sess,'lstm_classify_1.model',global_step=i))
 
 
train_lstm()
 
 
#————————————————预测模型————————————————————
'''
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)    
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('model')
        saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})  
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()
'''
