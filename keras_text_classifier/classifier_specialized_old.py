import sys
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

max_words = 2500
max_length = 35
EMBEDDING_DIM = 300
plot_history_check = True
gen_test = True

# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 8

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

filename = "new_train"
try:
    trainData = pd.read_csv("../data/"+filename+"_with_cname.csv")
    print("custom train data used")
except:
    print("cannot find custom data, generating...")
    trainData = pd.read_csv("../data/"+filename+".csv")
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
        trainData.to_csv(path_or_buf='../data/'+filename+'_with_cname.csv', index=False)
    except:
        trainData.to_csv(path_or_buf='train_with_cname.csv', index=False)


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

encoder_fashion = LabelEncoder()
encoder_fashion.fit(train_tags_fashion)
encoder_beauty = LabelEncoder()
encoder_beauty.fit(train_tags_beauty)
encoder_mobile = LabelEncoder()
encoder_mobile.fit(train_tags_mobile)

y_train_fashion = encoder_fashion.transform(train_tags_fashion)
y_train_beauty = encoder_beauty.transform(train_tags_beauty)
y_train_mobile = encoder_mobile.transform(train_tags_mobile)

# Converts the labels to a one-hot representation

y_train_fashion = utils.to_categorical(y_train_fashion, num_classes_fashion)
y_train_beauty = utils.to_categorical(y_train_beauty, num_classes_beauty)
y_train_mobile = utils.to_categorical(y_train_mobile, num_classes_mobile)


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

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting


def gen_filename_hdf5(name):
    filepath = "../checkpoints/" + name+'epoch_'+str(epochs) + '_' + \
               datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + "v2.hdf5"
    return filepath


checkpointer_fashion = ModelCheckpoint(gen_filename_hdf5("fashion"), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpointer_beauty = ModelCheckpoint(gen_filename_hdf5("beauty"), monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpointer_mobile = ModelCheckpoint(gen_filename_hdf5("mobile"), monitor='val_acc', verbose=1, save_best_only=True, mode='max')


history_fashion = model_fashion.fit([x_train_fashion], y_train_fashion,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=1,
                                    callbacks=[checkpointer_fashion],
                                    validation_split=0.1)

history_beauty = model_beauty.fit([x_train_beauty], y_train_beauty,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[checkpointer_beauty],
                                  validation_split=0.1)

history_mobile = model_mobile.fit([x_train_mobile], y_train_mobile,
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[checkpointer_mobile],
                                  validation_split=0.1)


def gen_filename_h5(history):
    return str(epochs) + '_' + str(max_words) + '_' + \
           str(history.history['val_acc'][-1]).replace('.', ',')[:5]


def gen_filename_csv():
    return 'epoch_'+str(epochs) + '_' + str(max_words) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


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
    plot_history(history_fashion)
    plot_history(history_beauty)
    plot_history(history_mobile)

# save model
model_fashion.save('model_fashion_' + gen_filename_h5(history_fashion) + '.h5')
model_beauty.save('model_beauty_' + gen_filename_h5(history_beauty) + '.h5')
model_mobile.save('model_mobile_' + gen_filename_h5(history_mobile) + '.h5')


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


# For plotting
def plotting(model, text_labels, x_validate, y_validate):
    y_softmax = model.predict(x_validate)

    y_test_1d = []
    y_pred_1d = []

    for i in range(len(y_validate)):
        probs = y_validate[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(24, 20))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    plt.show()
