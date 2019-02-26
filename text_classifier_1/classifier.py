import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sb

import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics
import json

import re

trainData = pd.read_csv("../data/train.csv")
dictData = pd.read_csv("../data/kata_dasar_kbbi.csv")
categories_file = open("../data/categories.json", "r")
categories = json.load(categories_file)
inverted_categories_mobile = {v: k.lower() for k, v in categories['Mobile'].items()}
inverted_categories_fashion = {v: k.lower() for k, v in categories['Fashion'].items()}
inverted_categories_beauty = {v: k.lower() for k, v in categories['Beauty'].items()}

category_mapping = {
    'fashion_image': 'Fashion',
    'beauty_image': 'Beauty',
    'mobile_image': 'Mobile',
}
directory_mapping = {
    'Fashion': 'fashion_image',
    'Beauty': 'beauty_image',
    'Mobile': 'mobile_image',
}

dic = {}
dicAll = {}

if os.path.isfile("./dic.pickle") and os.path.isfile("./dicAll.json"):
    with open("dic.pickle", "rb") as f:
        dic = pickle.load(f)
    with open("dicAll.json", "r") as f:
        dicAll = json.load(f)
    print("Trained data successfully loaded!")
else:
    print("Creating new data...")
    for index, row in trainData.iterrows():
        arr = re.split('\W+', row["title"])
        for sz in arr:
            if (sz, row["Category"]) not in dic:
                dic[(sz, row["Category"])] = 1
            else:
                dic[(sz, row["Category"])] += 1
            if sz not in dicAll:
                dicAll[sz] = 1
            else:
                dicAll[sz] += 1
    with open("dic.pickle", "wb+") as f:
        pickle.dump(dic, f)

    with open("dicAll.json", "w+") as f:
        json.dump(dicAll, f)


testData = pd.read_csv("../data/test.csv")
idx = []
res = []


it = 0
for index, row in testData.iterrows():
    maxi = 0.0
    imax = 0
    s = row["title"]
    img_path = row["image_path"]
    cat = category_mapping[img_path.split('/')[0]]
    arr = re.split('\W+', s)
    if cat == 'Fashion':
        sub_cats = inverted_categories_fashion
    elif cat == 'Mobile':
        sub_cats = inverted_categories_mobile
    elif cat == 'Beauty':
        sub_cats = inverted_categories_beauty
    for sub_cat_id, sub_cat_name in sub_cats.items():
        tot = 0.0

        # Subcat add score function
        if sub_cat_name.lower() in s.lower():
            tot += 1
        # Add score based on each substring in category
        for sz in arr:
            if (sz, sub_cat_id) in dic:
                tot += dic[(sz, sub_cat_id)] / dicAll[sz]
        if tot > maxi:
            maxi = tot
            imax = sub_cat_id
    idx.append(row["itemid"])
    res.append(imax)
    it += 1
    # if it > 10:
    #     break

df = pd.DataFrame({'itemid': idx, 'Category':res})
df.to_csv(path_or_buf='res.csv', index=False)

