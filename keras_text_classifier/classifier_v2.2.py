from datetime import datetime
import itertools
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import tensorflow_hub as hub
from keras.engine import Layer
from keras_preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, LSTM, \
    Bidirectional, CuDNNLSTM, MaxPooling1D, ConvLSTM2D, CuDNNGRU, SpatialDropout1D
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd
from keras import backend as K
from utility.train_data_loader import load_train_data

testData = pd.read_csv("../data/new_test.csv")
dictData = pd.read_csv("../data/kata_dasar_kbbi.csv")
categories_file = open("../data/categories.json", "r")
trainData = load_train_data()

# Shuffle train data
trainData = shuffle(trainData)

categories = json.load(categories_file)
inverted_categories_mobile = {v: k.lower() for k, v in categories['Mobile'].items()}
inverted_categories_fashion = {v: k.lower() for k, v in categories['Fashion'].items()}
inverted_categories_beauty = {v: k.lower() for k, v in categories['Beauty'].items()}

all_subcategories = {k.lower(): v for k, v in categories['Mobile'].items()}
all_subcategories.update({k.lower(): v for k, v in categories['Fashion'].items()})
all_subcategories.update({k.lower(): v for k, v in categories['Beauty'].items()})

# Main settings
plot_history_check = True
gen_test = False
max_length = 35  # 32 is max word in train
max_words = 2500
EMBEDDING_DIM = 300  # Based on the txt file: glove 300d
num_classes = len(all_subcategories)
# Training for more epochs will likelval-acc after 10 epochs: 0.71306y lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 10

print(all_subcategories)
print("no of categories: " + str(num_classes))

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

max_data_size = int(len(trainData) * 1)
train_data_size = int(max_data_size * .95)
train_data_step = 1
validate_data_step = 1
print(train_data_size, max_data_size)

train_texts = trainData['title'][::train_data_step]
train_tags = trainData['Category'][::train_data_step]
test_texts = testData['title']
print(len(train_texts), len(train_tags))

y_train = train_tags.values
y_train = utils.to_categorical(y_train)

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)


class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                           as_dict=True,
                           signature='default',
                           )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)

# Val-acc after 10 epochs: 0.7068
model = Sequential()
model.add(ElmoEmbeddingLayer(input_shape=(1,), input_dtype="string"))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

model.fit(train_texts, y=y_train, batch_size=batch_size, validation_split=0.1, verbose=1,
          shuffle=True, epochs=epochs)


def gen_filename_csv():
    return 'epoch_'+str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


def perform_test():
    prediction = model.predict(test_texts, batch_size=batch_size, verbose=1)
    predicted_label = [np.argmax(prediction[i]) for i in range(len(test_texts))]
    # print(predicted_label)
    df = pd.DataFrame({'itemid': testData['itemid'].astype(int), 'Category': predicted_label})
    df.to_csv(path_or_buf='res_' + gen_filename_csv() + '.csv', index=False)


if gen_test:
    perform_test()