import numpy as np
import pandas as pd

def dataConvert():
    df=pd.read_excel("svm_train.xlsx",header=None)
    label=[]
    s=[]
    m=int(409600/1024)
    for j in range(m):
        df1=df.iloc[j*1024:(j+1)*1024,0:5]  
        df1=df1.values
        s1=[]
        s2=[]
        s3=[]
        s4=[]
        print(j)
        for i in range(len(df1)):
            s1.append(df1[i][0])
            s2.append(df1[i][1])
            s3.append(df1[i][2])
            s4.append(df1[i][3])
        x=1 if df1[0][4]!=0 else 0
        label.append(x)
        label.append(x)
        label.append(x)
        label.append(x)    
        s.append(s1)
        s.append(s2)
        s.append(s3)
        s.append(s4)
    print(len(s))
    print(len(label))
    np.save("svm_train.npy",s)
    np.save("svm_label.npy",label)
dataConvert()

        