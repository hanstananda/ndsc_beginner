import sys
from datetime import datetime
import itertools
import json
import subprocess
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.sequence import pad_sequences

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, CuDNNLSTM, \
    SpatialDropout1D, MaxPooling1D, Flatten, BatchNormalization
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd

sys.setrecursionlimit(10000)

testData = pd.read_csv("../data/new_test.csv")
dictData = pd.read_csv("../data/kata_dasar_kbbi.csv")
categories_file = open("../data/categories.json", "r")
categories = json.load(categories_file)
inverted_categories_mobile = {v: k.lower() for k, v in categories['Mobile'].items()}
num_classes_mobile = len(inverted_categories_mobile)
inverted_categories_fashion = {v: k.lower() for k, v in categories['Fashion'].items()}
num_classes_fashion = len(inverted_categories_fashion)
inverted_categories_beauty = {v: k.lower() for k, v in categories['Beauty'].items()}
num_classes_beauty = len(inverted_categories_beauty)

all_subcategories = {k.lower(): v for k, v in categories['Mobile'].items()}
all_subcategories.update({k.lower(): v for k, v in categories['Fashion'].items()})
all_subcategories.update({k.lower(): v for k, v in categories['Beauty'].items()})


def update_embeddings_index():
    embeddings_index = {}
    for line in glove_file:
        values = line.split()
        word = ''.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        # print(coefs)
        embeddings_index[word] = coefs
    return embeddings_index


try:
    print("using glove data from joblib...")
    embeddings_index = joblib.load("../data/glove.840B.300d.joblib")
    print("glove data loaded from joblib!")
except:
    print("using glove data from txt...")
    glove_file = open('../data/glove.840B.300d.txt', "r", encoding="Latin-1")
    embeddings_index = update_embeddings_index()
    print("glove data loaded from txt!")
    joblib.dump(embeddings_index, "../data/glove.840B.300d.joblib")
    print("glove data saved to joblib!")


# Main settings

max_words = 5000
max_length = 35
EMBEDDING_DIM = 300
plot_history_check = True
gen_test = True
submit = False

# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 9

print(all_subcategories)
print("no of categories: " + str(len(all_subcategories)))

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

test_data_fashion = testData[testData['image_path'].str.contains("fashion")]
test_data_beauty = testData[testData['image_path'].str.contains("beauty")]
test_data_mobile = testData[testData['image_path'].str.contains("mobile")]

change_fashion = False
change_beauty = False
change_mobile = False

source_file_name = ""
dest_file_name = ""

sourceDF = pd.read_csv(source_file_name)
destDF = pd.read_csv(dest_file_name)

if change_fashion:
    for keys,rows in test_data_fashion:
        destDF['itemid'].iloc[rows['itemid']] = sourceDF['itemid'].iloc[rows['itemid']]

if change_beauty:
    for keys,rows in test_data_beauty:
        destDF['itemid'].iloc[rows['itemid']] = sourceDF['itemid'].iloc[rows['itemid']]

if change_mobile:
    for keys,rows in test_data_mobile:
        destDF['itemid'].iloc[rows['itemid']] = sourceDF['itemid'].iloc[rows['itemid']]

