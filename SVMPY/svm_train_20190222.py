# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:25:58 2018

@author: wangzhijie
"""
import pandas as pd
import xlrd
import os
import numpy as np
strPath="D:\\matlab\\install\\bin\\project\\esmd4matlab_1.0\\tezhengxiangliang"
l=os.listdir(strPath)
svm_train=[]
writer=pd.ExcelWriter("svm_train.xlsx");
for i in l:
    print(i+"---------------------------------------")
    #读取每个excel文件中前40个
    labels=[0]
    if "B" in i:
        labels=[1]
    elif "IR" in i:
        labels=[2]
    elif "OR" in i:
        labels=[3]
    else:
        labels=[0]
    for j in range(1,41): 
        data=[]
        print("Sheet"+str(j))
        data1=pd.read_excel(strPath+"\\"+i,"Sheet"+str(j),header=None)
        data1[data1.columns.size]=np.array(labels).repeat(data1.iloc[:,0].size)
        for z in range(len(data1)): 
            svm_train.append(data1.loc[z])  
pd.DataFrame(svm_train).to_excel(excel_writer=writer,header=False,index=None,sheet_name="Sheet1")
writer.save()
writer.close()