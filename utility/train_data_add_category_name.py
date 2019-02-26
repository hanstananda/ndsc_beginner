import pandas as pd
import json

trainData = pd.read_csv("../data/train.csv")
testData = pd.read_csv("../data/test.csv")
dictData = pd.read_csv("../data/kata_dasar_kbbi.csv")
categories_file = open("../data/categories.json", "r")
categories = json.load(categories_file)
inverted_categories_mobile = {v: k.lower() for k, v in categories['Mobile'].items()}
inverted_categories_fashion = {v: k.lower() for k, v in categories['Fashion'].items()}
inverted_categories_beauty = {v: k.lower() for k, v in categories['Beauty'].items()}

all_subcategories = {k.lower(): v for k, v in categories['Mobile'].items()}
all_subcategories.update({k.lower(): v for k, v in categories['Fashion'].items()})
all_subcategories.update({k.lower(): v for k, v in categories['Beauty'].items()})

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

