import json

import pandas as pd

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


def load_train_data():
    try:
        trainData = pd.read_csv("../data/train_with_cname.csv")
        print("custom train data used")
    except:
        print("cannot find custom data, generating...")
        trainData = pd.read_csv("../data/train.csv")
        trainData['item_category'] = 'None'
        for index, row in trainData.iterrows():
            s = row["title"]
            img_path = row["image_path"]
            cat = category_mapping[img_path.split('/')[0]]
            if cat == 'Fashion':
                sub_cats = inverted_categories_fashion
            elif cat == 'Mobile':
                sub_cats = inverted_categories_mobile
            elif cat == 'Beauty':
                sub_cats = inverted_categories_beauty
            # trainData.set_value(index, 'item_category', sub_cats[row['Category']])
            trainData.at[index, 'item_category'] = sub_cats[row['Category']]
        try:
            trainData.to_csv(path_or_buf='../data/train_with_cname.csv', index=False)
        except:
            trainData.to_csv(path_or_buf='train_with_cname.csv', index=False)
    return trainData

