# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 20:05:44 2018

@author: wangzhijie
"""

import pandas as pd
import numpy as np
import xlrd
import os
time_step=32
writer=pd.ExcelWriter("lstm_train_1.xlsx");
data=pd.read_excel("lstm_train.xlsx",sheet_name="Sheet1",header=None)
label=0;
train_data=[];
for i in range(len(data)):
    print(str(i+1)+"行")
    if i>=0 and i<=119:
        label=[1,0,0,0]
    elif i>=120 and i<240:
        label=[0,1,0,0]
    elif i>=240 and i<280:
        label=[0,0,1,0]
    else:
        label=[0,0,0,1]
    for j in range(int(1024/time_step)):
        a=[]
        a.extend(data.iloc[i,time_step*j*4:time_step*(j+1)*4])
        a.extend(label)        
        train_data.append(a)
        print(str(len(train_data))+" "+str(len(train_data[len(train_data)-1])))        
pd.DataFrame(train_data).to_excel(excel_writer=writer,header=False,index=None,sheet_name="Sheet1")
writer.save()
writer.close()