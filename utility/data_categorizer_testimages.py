import json
import os
from shutil import copyfile
import pandas as pd


categories_file = open("../data/categories.json", "r")
categories = json.load(categories_file)
test_data = pd.read_csv("../data/test.csv")
source_parent_path = '../../'
destination_path = '../../'
directory_mapping = {
    'Fashion': 'fashion_image',
    'Beauty': 'beauty_image',
    'Mobile': 'mobile_image',
}
category = 'Beauty'
limit = 100000

# Find all image files
rows = test_data[test_data['image_path'].str.contains(directory_mapping[category], na=False)]
files = rows['image_path'].tolist()
print(len(files))


subcategory = files[0].split('/')[0]
if not os.path.exists(os.path.join(destination_path, subcategory, "test")):
    os.mkdir(os.path.join(destination_path, subcategory, "test"))
    print("New folder created!")
counter = 0
for fl in files:
    if ".jpg" not in fl:  # fix typos
        fl = fl+".jpg"
    src = os.path.join(source_parent_path, fl)
    tmp = fl.split('/')
    des = os.path.join(destination_path, tmp[0], "test", tmp[1])
    # print(des)
    copyfile(src, des)
    # print(src)
    counter += 1
    if counter > limit:
        break

print(counter)

