from datetime import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, LSTM, \
    Bidirectional, MaxPooling1D, TimeDistributed, SpatialDropout1D, CuDNNLSTM
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd

from utility.train_data_loader import load_train_data

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

# Main settings
plot_history_check = True
gen_test = True
max_length = 35  # 32 is max word in train
max_words = 2500
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

trainData = load_train_data()

# Shuffle train data
trainData = shuffle(trainData)

max_data_size = int(len(trainData) * 1)
train_data_size = int(max_data_size * .95)
train_data_step = 1
validate_data_step = 1
print(train_data_size, max_data_size)

train_texts = trainData['title'][::train_data_step]
train_tags = trainData['Category'][::train_data_step]
test_texts = testData['title']
print(len(train_texts), len(train_tags))

y = train_tags.values

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_texts)  # only fit on train
x_train = tokenize.texts_to_sequences(train_texts)
x_test = tokenize.texts_to_sequences(test_texts)

word_index = tokenize.word_index

# Pad sequences with zeros
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

y_train = train_tags.values
y_train = utils.to_categorical(y_train)
vocab_size = len(tokenize.word_index) + 1
print(vocab_size)

# Tested val-acc parameter:
# max_length = 35, max_words = 1000

# model 1 : Embedding with normal Dense NN Softmax
# max val-acc after 10 epochs: 0.70347
# max val-acc after 50 epochs: 0.70442

# model = Sequential()
# model.add(Embedding(len(word_index)+1,
#                     300,
#                     input_length=max_length,
#                     trainable=True))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# model 2 : Embedding with LSTM RNN
# max val-acc after 10 epochs: 0.71602
# max val-acc after 50 epochs: 0.71978
# Additional note: The training time is freaking longer than others, more than 3 times model 1!

# model = Sequential()
# model.add(Embedding(len(word_index)+1,
#                     300,
#                     input_length=max_length,
#                     trainable=True))
# model.add(Bidirectional(LSTM(100)))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()


# model 3 : Embedding with Convolutional NN
# val-acc after 10 epochs: 0.71306
# val-acc after 50 epochs: 0.71369

# model = Sequential()
# model.add(Embedding(len(word_index)+1,
#                     300,
#                     input_length=max_length,
#                     trainable=True))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#model.summary()



# model 3.1 : Embedding with multilevel CNN
# Note: Max val acc after 50epoch is 0.70544, converges alrd
# Note: This one very long time to train, max length must also be 1000++

# model = Sequential()
# model.add(Embedding(len(word_index)+1,
#                     300,
#                     input_length=max_length,
#                     trainable=True))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(MaxPooling1D(5))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()


# model 4
# Note: Max val after 10 epochs: Not tested, but clearly better than CUDNNLSTM normal XD

# model = Sequential()
# model.add(Embedding(len(word_index)+1,
#                     300,
#                     input_length=max_length,
#                     trainable=True))
# model.add(SpatialDropout1D(0.2))
# model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
# model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
# model.add(Conv1D(256, 5, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()

# model 4.1
# Note: Max val after 10 epochs: 0.734 (may still slightly increase xD)

# model = Sequential()
# model.add(Embedding(len(word_index)+1,
#                     300,
#                     input_length=max_length,
#                     trainable=True))
# model.add(Dropout(0.25))
# model.add(Conv1D(256, 5, activation='relu', padding='valid', strides=1))
# model.add(MaxPooling1D(pool_size=4))
# model.add(CuDNNLSTM(256))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.summary()


# model 4.2
# 0.724 in 6 epochs, a bit worse than model 4

model = Sequential()
model.add(Embedding(len(word_index)+1,
                    300,
                    input_length=max_length,
                    trainable=True))
model.add(Dropout(0.25))
model.add(TimeDistributed(Conv1D(256, 5, activation='relu', padding='same', strides=1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
model.add(TimeDistributed(Conv1D(256, 5, activation='relu', padding='same', strides=1)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
model.add(Bidirectional(CuDNNLSTM(128, return_sequences=False)))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()



def gen_filename_h5():
    return 'epoch_'+str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


def gen_filename_csv():
    return 'epoch_'+str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# Checkpoint auto
filepath = "../checkpoints/"+gen_filename_h5()+"v2.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

history = model.fit([x_train], batch_size=batch_size, y=y_train, verbose=1, validation_split=0.1,
                    shuffle=True, epochs=epochs, callbacks=[checkpointer])


def plot_history(history):
    plt.style.use('ggplot')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if plot_history_check:
    plot_history(history)


def perform_test():
    prediction = model.predict(x_test, batch_size=batch_size, verbose=1)
    predicted_label = [np.argmax(prediction[i]) for i in range(len(x_test))]
    # print(predicted_label)
    df = pd.DataFrame({'itemid': testData['itemid'].astype(int), 'Category': predicted_label})
    df.to_csv(path_or_buf='res_' + gen_filename_csv() + '.csv', index=False)


if gen_test:
    perform_test()


# This utility function is from the sklearn docs:
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)
    plt.show()
