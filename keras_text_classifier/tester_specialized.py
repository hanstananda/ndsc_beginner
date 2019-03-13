from utility.train_data_loader import load_train_data
from datetime import datetime
import itertools
import json

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
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Bidirectional, CuDNNLSTM
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd

max_words = 2500
max_length = 35
EMBEDDING_DIM = 300
batch_size = 256
epochs = 10
gen_test = True


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


testData = pd.read_csv("../data/new_test.csv")
trainData = load_train_data()

train_data_fashion = trainData[trainData['image_path'].str.contains("fashion")]
train_data_beauty = trainData[trainData['image_path'].str.contains("beauty")]
train_data_mobile = trainData[trainData['image_path'].str.contains("mobile")]
test_data_fashion = testData[testData['image_path'].str.contains("fashion")]
test_data_beauty = testData[testData['image_path'].str.contains("beauty")]
test_data_mobile = testData[testData['image_path'].str.contains("mobile")]

# Shuffle train data
train_data_fashion = shuffle(train_data_fashion)
train_data_beauty = shuffle(train_data_beauty)
train_data_mobile = shuffle(train_data_mobile)

train_texts_fashion = train_data_fashion['title']
train_texts_beauty = train_data_beauty['title']
train_texts_mobile = train_data_mobile['title']
test_texts_fashion = test_data_fashion['title']
test_texts_beauty = test_data_beauty['title']
test_texts_mobile = test_data_mobile['title']


train_tags_fashion = train_data_fashion['item_category']
train_tags_beauty = train_data_beauty['item_category']
train_tags_mobile = train_data_mobile['item_category']

tokenize_fashion = text.Tokenizer(num_words=max_words, char_level=False)
tokenize_fashion.fit_on_texts(train_texts_fashion)
tokenize_beauty = text.Tokenizer(num_words=max_words, char_level=False)
tokenize_beauty.fit_on_texts(train_texts_beauty)
tokenize_mobile = text.Tokenizer(num_words=max_words, char_level=False)
tokenize_mobile.fit_on_texts(train_texts_mobile)

x_train_fashion = tokenize_fashion.texts_to_sequences(train_texts_fashion)
x_train_beauty = tokenize_beauty.texts_to_sequences(train_texts_beauty)
x_train_mobile = tokenize_mobile.texts_to_sequences(train_texts_mobile)
x_test_fashion = tokenize_fashion.texts_to_sequences(test_texts_fashion)
x_test_beauty = tokenize_beauty.texts_to_sequences(test_texts_beauty)
x_test_mobile = tokenize_mobile.texts_to_sequences(test_texts_mobile)

# Pad sequences with zeros
x_train_fashion = pad_sequences(x_train_fashion, padding='post', maxlen=max_length)
x_train_beauty = pad_sequences(x_train_beauty, padding='post', maxlen=max_length)
x_train_mobile = pad_sequences(x_train_mobile, padding='post', maxlen=max_length)
x_test_fashion = pad_sequences(x_test_fashion, padding='post', maxlen=max_length)
x_test_beauty = pad_sequences(x_test_beauty, padding='post', maxlen=max_length)
x_test_mobile = pad_sequences(x_test_mobile, padding='post', maxlen=max_length)

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


encoder_fashion = LabelEncoder()
encoder_fashion.fit(train_tags_fashion)
encoder_beauty = LabelEncoder()
encoder_beauty.fit(train_tags_beauty)
encoder_mobile = LabelEncoder()
encoder_mobile.fit(train_tags_mobile)

word_index_fashion = tokenize_fashion.word_index
word_index_beauty = tokenize_beauty.word_index
word_index_mobile = tokenize_mobile.word_index


def get_embedding_matrix(word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


embedding_matrix_fashion = get_embedding_matrix(word_index_fashion)
embedding_matrix_beauty = get_embedding_matrix(word_index_beauty)
embedding_matrix_mobile = get_embedding_matrix(word_index_mobile)


# Build the model
def model_gen(num_classes, word_index, embedding_matrix):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        300,
                        input_length=max_length,
                        weights=[embedding_matrix],
                        trainable=True))
    model.add(Bidirectional(CuDNNLSTM(256, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(256)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


model_fashion = model_gen(num_classes_fashion, word_index_fashion, embedding_matrix_fashion)
model_beauty = model_gen(num_classes_beauty, word_index_beauty, embedding_matrix_beauty)
model_mobile = model_gen(num_classes_mobile, word_index_mobile, embedding_matrix_mobile)


def gen_filename_csv():
    return 'epoch_'+str(epochs) + '_' + str(max_words) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


model_fashion.load_weights('../checkpoints/fashionepoch_10_03_13_2019_02_39_49v2.hdf5')
model_beauty.load_weights('../checkpoints/beautyepoch_10_03_13_2019_02_39_49v2.hdf5')
model_mobile.load_weights('../checkpoints/mobileepoch_10_03_13_2019_02_39_49v2.hdf5')


def perform_test():
    prediction_fashion = model_fashion.predict(x_test_fashion, batch_size=batch_size, verbose=1)
    prediction_beauty = model_beauty.predict(x_test_beauty, batch_size=batch_size, verbose=1)
    prediction_mobile = model_mobile.predict(x_test_mobile, batch_size=batch_size, verbose=1)
    predicted_label_fashion = [all_subcategories[encoder_fashion.classes_[np.argmax(prediction_fashion[i])]]
                               for i in range(len(x_test_fashion))]
    predicted_label_beauty = [all_subcategories[encoder_beauty.classes_[np.argmax(prediction_beauty[i])]]
                              for i in range(len(x_test_beauty))]
    predicted_label_mobile = [all_subcategories[encoder_mobile.classes_[np.argmax(prediction_mobile[i])]]
                              for i in range(len(x_test_mobile))]

    df = pd.DataFrame({'itemid': test_data_fashion['itemid'].astype(int), 'Category': predicted_label_fashion})
    df = df.append(pd.DataFrame({'itemid': test_data_beauty['itemid'].astype(int), 'Category': predicted_label_beauty}))
    df = df.append(pd.DataFrame({'itemid': test_data_mobile['itemid'].astype(int), 'Category': predicted_label_mobile}))
    # print(predicted_label_fashion)
    # print(prediction_beauty)
    # print(prediction_mobile)

    # for i, row in testData.iterrows():
    #     prediction = model.predict(np.array([x_test[i]]))
    #     predicted_label = text_labels[np.argmax(prediction)]
    #     label_id = all_subcategories[predicted_label]
    #     indexes.append(row["itemid"])
    #     results.append(label_id)
    #
    # df = pd.DataFrame({'itemid': indexes, 'Category': results})
    df.to_csv(path_or_buf='res' + gen_filename_csv() + '.csv', index=False)


if gen_test:
    perform_test()