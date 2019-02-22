import pandas as pd
import xlrd
import os
strPath="D:\\matlab\\install\\bin\\project\\esmd4matlab_1.0\\tezhengxiangliang"
l=os.listdir(strPath)
lstm_train=[]
writer=pd.ExcelWriter("test_lstm.xlsx");
for i in l:
    print(i+"---------------------------------------")
    #读取每个excel文件中前40个
    labels=[0,0,0,0]
    if "B" in i:
        labels=[1,0,0,0]
    elif "IR" in i:
        labels=[0,1,0,0]
    elif "OR" in i:
        labels=[0,0,1,0]
    else:
        labels=[0,0,0,1]
    for j in range(41,51): 
        data=[]
        print("Sheet"+str(j))
        data1=pd.read_excel(strPath+"\\"+i,"Sheet"+str(j),header=None)
        #转型
        for z in range(1,data1.shape[0]+1):
            data.extend(data1.loc[z-1])
        data.extend(labels)
        lstm_train.append(data)
pd.DataFrame(lstm_train).to_excel(excel_writer=writer,header=False,index=None,sheet_name="Sheet1")
writer.save()
writer.close()
