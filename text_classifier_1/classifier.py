import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics
import json

import re

trainData = pd.read_csv("../data/train.csv")
dictData = pd.read_csv("../data/kata_dasar_kbbi.csv")

dic = {}
dicAll = {}

for index,row in trainData.iterrows():
    arr = re.split('\W+',row["title"])
    for sz in arr:
        if (sz,row["Category"]) not in dic:
            dic[(sz,row["Category"])] = 1
        else:
            dic[(sz,row["Category"])]+=1

for index,row in trainData.iterrows():
    arr = re.split('\W+',row["title"])
    for sz in arr:
        if sz not in dicAll:
            dicAll[sz] = 1
        else:
            dicAll[sz]+=1

dicAll = {}

with open("dic.json", "w+") as f:
    json.dump(dic, f)

with open("dicAll.json", "w+") as f:
    json.dump(dicAll, f)


testData = pd.read_csv("test.csv")
idx = []
res = []

for index,row in testData.iterrows():
    maxi = 0.0
    imax = 0
    for i in range(58):
        tot = 0.0
        arr = re.split('\W+',row["title"])
        for sz in arr:
            if (sz,i) in dic:
                tot += dic[(sz,i)]/dicAll[sz]
        if tot>maxi:
            maxi=tot
            imax = i
    idx.append(row["itemid"])
    res.append(imax)
df = pd.DataFrame({'itemid': idx, 'Category':res})
df.to_csv(path_or_buf  = 'res.csv', index = False)

