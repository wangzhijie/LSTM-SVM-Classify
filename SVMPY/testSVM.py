from svm import BaggingSVM
import numpy as np

#把自己的数据放入里面试着跑下数据
#如果不行就还用这个数据，然后将数据放入BP中跑下
#如果可以，就将数据放入BP中跑下
#然后将数据放入LSTM中跑


def createDataSet():
    # # 生成一个矩阵，每行表示一个样本
    # group = array([[1.0, 0.9], [1.0, 1.0], [0.1, 0.2], [0.0, 0.1], [1.0, 0.8], [1.0, 1.1], [0.1, 0.1], [0.0, 0.2],
    #                [1.0, 0.7], [1.0, 1.1], [0.1, 0.3], [0.1, 0.1], [1.0, 0.6], [1.0, 1.2], [0.1, 0.2], [0.1, 0.3], [0.1, 0.2], [0.1, 0.3]])
    # # 4个样本分别所属的类别
    # labels =array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    group=np.load('svm_train.npy')    # 样本（训练集）
    labels=np.load('svm_label.npy')   # 样本标签（训练集）
  #  labels=np.ones(40)
 #   labels=np.append(labels,np.zeros(360))
    return group,labels
x,y = createDataSet()
print(x)
print(y)
W = BaggingSVM(100, 150, 1)

weight=np.ones(len(x))
W.fit(x,y,weight)  #训练分类器

pre_x=x[0:300,:]
pre_lab=y[0:300]
pre=W.predict(pre_x)
print("===========================================================")
#print(np.load('TestLabel_SMOTE75.npy'))
ture=0
fa=0
for i in range(len(pre_x)):
    if pre_lab[i] == pre[i] :
        print(i)
        ture=ture+1
    else :
        fa=fa+1
        print(i)

print('t',ture)
print('f',fa)
