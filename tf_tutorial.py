import json
import pandas as pd

import dataset

train_data = pd.read_csv("train.csv")
train_data.sort_values('Category')
print(train_data.head(5))
test_data = pd.read_csv("test.csv")
categories_file = open("categories.json", "r")
categories = json.load(categories_file)
beauty_subcategory = categories['Beauty']

classes = []

# for key, value in beauty_subcategory.items():
#     classes.append(value)

for names in beauty_subcategory:
    classes.append(names)

print(classes)

num_classes = len(classes)

train_path = '../'

# validation split
validation_size = 0.2

# batch size
batch_size = 16

img_size = 100


data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)