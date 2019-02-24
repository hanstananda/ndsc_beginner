import json

import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import PIL.Image as Image

import pandas as pd

KBBI = pd.read_csv("kata_dasar_kbbi.csv")
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

categories_file = open("categories.json", "r")
categories = json.load(categories_file)
mobile_subcategory = categories['Mobile']
fashion_subcategory = categories['Fashion']
beauty_subcategory = categories['Beauty']

print(type(mobile_subcategory))

print(KBBI.head(5))
print(train_data.head(5))