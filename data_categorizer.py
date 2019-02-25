import json
import os
from shutil import copyfile
import pandas as pd


categories_file = open("categories.json", "r")
categories = json.load(categories_file)
train_data = pd.read_csv("train.csv")
train_data.sort_values('Category')
train_path = '../'
classes = []


def get_category_id(class_name):
    return categories['Mobile'][class_name]

def get_files(fid):
    rows = train_data.loc[train_data['Category'] == fid]
    files = rows['image_path'].tolist()
    return files


for names in categories['Mobile']:
    classes.append(names)

for fields in classes:
    index = classes.index(fields)
    print('Now going to read {} files (Index: {})'.format(fields, index))
    files = get_files(get_category_id(fields))
    subcategory = files[0].split('/')[0]
    if not os.path.exists(os.path.join(train_path, subcategory, fields)):
        os.mkdir(os.path.join(train_path, subcategory, fields))
        print("New folder created!")
    counter = 0
    for fl in files:
        src = os.path.join(train_path, fl)
        tmp = fl.split('/')
        des = os.path.join(train_path, tmp[0], fields, tmp[1])
        # print(des)
        copyfile(src,des)
        # print(src)
        counter += 1

    print(counter)

