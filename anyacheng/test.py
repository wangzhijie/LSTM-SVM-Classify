# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:59:31 2019

@author: wangzhijie
"""

import pandas as pd

df=pd.read_csv("datawavlet.csv")
print(df.iloc[1:2,:])