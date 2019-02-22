# 打开师姐经过小波包处理的CSV格式数据()
import csv
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
#import WeightSMOTE

# filename = 'D:/py/datawavlet.csv'
'''
filename = 'C:/Users/曹雪/Desktop/shuju/data1.csv'
with open(filename) as f:
    reader = csv.reader(f)
    data=list(reader)
    # print(data[1])
datanp=np.array(data)
'''

#打开datatest
filenameTest = 'datawavlet.csv'
with open(filenameTest) as f:
    reader = csv.reader(f)
    datatest=list(reader)

datatestnp=np.array(datatest)
print('len(datatestnp)',len(datatestnp))
'''
print(len(datanp))
tdata=np.array([])
traindata=np.array([])
for i in range(1,len(datanp)):
    tdata=np.append(tdata,datanp[i])
for i in range(len(datanp)-1):
    traindata = np.append(traindata, tdata[i].split())

    traindata = traindata.astype(np.float64)

    traindata=traindata.reshape(-1,2048)
print(len(traindata))
neigh = NearestNeighbors(6)
neigh.fit(traindata)
print(traindata.shape[0])
print(neigh.kneighbors(traindata ,6 ,return_distance=False))
'''
#datatest
tedata=np.array([])
testdata=np.array([])
for i in range(1,len(datatestnp)):
    print(i)
    tedata=np.append(tedata,datatestnp[i])
for i in range(len(datatestnp)-1):
    print(i)
    testdata = np.append(testdata, tedata[i].split())

    testdata = testdata.astype(np.float64)

    testdata=testdata.reshape(-1,2048)
print(len(testdata))
np.save("svm_train.npy",testdata)

'''
neigh = NearestNeighbors(6)
neigh.fit(testdata)
print(testdata.shape[0])
print(neigh.kneighbors(testdata ,6 ,return_distance=False))
'''
'''
bs=WeightSMOTE.WeightSMOTE( )
label=np.array([])
for i in range (0,100):
    label = np.append(label, 1)
for i in range(100,180):
    label = np.append(label, 0)
print('wwww',len(label))
traindata_SMOTE, label_SMOTE,weight_SMOTE=bs.SMOTE(6,traindata,label)

print('traindata_SMOTE',len(traindata_SMOTE))
print('label_SMOTE',len(label_SMOTE))
print('weight_SMOTE',weight_SMOTE)
'''

# np.save("traindata_SMOTE180.npy", traindata_SMOTE)
# # 训练集1（样本）
# np.save('label_SMOTE180.npy',label_SMOTE)
# # 训练集1（标签）
# np.save('weight_SMOTE180.npy',weight_SMOTE)
# # 训练集1（SMOTE后权重）



# testdata
'''
bt=WeightSMOTE.WeightSMOTE( )
testlabel=np.array([])
for i in range (0,50):
    testlabel = np.append(testlabel, 1)
for i in range(50,75):
    testlabel = np.append(testlabel, 0)
testdata_SMOTE, testlabel_SMOTE,testweight_SMOTE=bt.SMOTE(6,testdata,testlabel)

# 对测试集样本进行过采样处理

print('testdata_SMOTE',len(testdata_SMOTE))
print('testlabel_SMOTE',testlabel_SMOTE)
print('testweight_SMOTE',testweight_SMOTE)
'''

# np.save("TestData_SMOTE75.npy", testdata_SMOTE)
# # # 测试集（样本）
# # np.save('TestLabel_SMOTE75.npy',testlabel_SMOTE)
# # # 测试集（标签）
# # np.save('Testeight_SMOTE75.npy',testweight_SMOTE)
# # # 测试集（权重）