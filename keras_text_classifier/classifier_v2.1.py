from datetime import datetime
import itertools
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from keras_preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, LSTM, \
    Bidirectional, CuDNNLSTM, MaxPooling1D, ConvLSTM2D, CuDNNGRU
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd
from utility.train_data_loader import load_train_data

testData = pd.read_csv("../data/new_test.csv")
dictData = pd.read_csv("../data/kata_dasar_kbbi.csv")
categories_file = open("../data/categories.json", "r")


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
max_words = 1000
EMBEDDING_DIM = 300  # Based on the txt file: glove 300d
num_classes = len(all_subcategories)
# Training for more epochs will likelval-acc after 10 epochs: 0.71306y lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 15

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



tokenize = text.Tokenizer(num_words=1000, char_level=False)
tokenize.fit_on_texts(train_texts)  # only fit on train
x_train = tokenize.texts_to_sequences(train_texts)
x_test = tokenize.texts_to_sequences(test_texts)

# Pad sequences with zeros
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

y_train = train_tags.values
y_train = utils.to_categorical(y_train)
word_index = tokenize.word_index


embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Tested val-acc parameter:
# max_length = 35, max_words = 1000

# model 1 : Embedding with normal Dense NN Softmax
# max val-acc after 10 epochs: 0.71885

model = Sequential()
model.add(Embedding(len(word_index) + 1,
                    300,
                    input_length=max_length,
                    weights=[embedding_matrix],
                    trainable=True))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# model 2 : Embedding with LSTM RNN
# max val-acc after 10 epochs: 0.72440
# Additional note: The training time is freaking longer than others w/o CUDNNLSTM
# Also, the relu dense layer is not used for now
# Note: Use LSTM if u want to use CPU

# model = Sequential()
# model.add(Embedding(len(word_index) + 1,
#                     300,
#                     input_length=max_length,
#                     weights=[embedding_matrix],
#                     trainable=True))
# model.add(Bidirectional(CuDNNLSTM(100)))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()



# model 2.1
# max val-acc after 10 epochs: 0.73003

# model = Sequential()
# model.add(Embedding(len(word_index) + 1,
#                     300,
#                     input_length=max_length,
#                     weights=[embedding_matrix],
#                     trainable=True))
# model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))
# model.add(Bidirectional(CuDNNLSTM(128)))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()

# model 2.2
# max val-acc after 15 epochs: 0.72782 (converges then go down after that)

# model = Sequential()
# model.add(Embedding(len(word_index) + 1,
#                     300,
#                     input_length=max_length,
#                     weights=[embedding_matrix],
#                     trainable=True))
# model.add(Bidirectional(CuDNNGRU(128, return_sequences=True)))
# model.add(Bidirectional(CuDNNGRU(128)))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()

# model 3 : Embedding with Convolutional NN
# val-acc after 10 epochs: 0.71
# Note : seems a bit less likely to increase

# model = Sequential()
# model.add(Embedding(len(word_index) + 1,
#                     300,
#                     input_length=max_length,
#                     weights=[embedding_matrix],
#                     trainable=True))
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.summary()


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
