from datetime import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd

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
maxlen = 50 # 32 is max word in train
num_classes = len(all_subcategories)
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 1
embedding_dim = 100
model_no = 2

print(all_subcategories)
print("no of categories: "+str(num_classes))

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


trainData = pd.read_csv("../data/train.csv")

# Shuffle train data
trainData = shuffle(trainData)

max_data_size = int(len(trainData) * .1)
train_data_size = int(max_data_size * .95)
train_data_step = 1
validate_data_step = 1
print(train_data_size, max_data_size)

train_texts = trainData['title'][:train_data_size:train_data_step]
train_tags = trainData['Category'][:train_data_size:train_data_step]
validate_texts = trainData['title'][train_data_size:max_data_size:validate_data_step]
validate_tags = trainData['Category'][train_data_size:max_data_size:validate_data_step]
test_texts = testData['title']
print(len(train_texts), len(train_tags))
print(len(validate_texts), len(validate_tags))


# Regression test
def regression():
    vectorizer = CountVectorizer()
    vectorizer.fit(train_texts)
    X_train = vectorizer.transform(train_texts)
    X_validate  = vectorizer.transform(validate_texts)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    score = classifier.score(X_validate, y_validate)
    print('Accuracy for {} data: {:.4f}'.format("data", score))
# regression()


max_words = 2500
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_texts)  # only fit on train
x_train = tokenize.texts_to_sequences(train_texts)
x_validate = tokenize.texts_to_sequences(validate_texts)
x_test = tokenize.texts_to_sequences(test_texts)

# Pad sequences with zeros
x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
x_validate = pad_sequences(x_validate, padding='post', maxlen=maxlen)
x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

y_train = train_tags.values
y_validate = validate_tags.values

print(len(y_train), len(y_validate))

vocab_size = len(tokenize.word_index) + 1
# Build the model

def model2():
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
                       output_dim=embedding_dim,
                       input_length=maxlen,
                        trainable=True))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def model3():
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim,
                               input_length=maxlen,
                               trainable=True))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if model_no == 2:
    model = model2()
elif model_no == 3:
    model = model3()
model.summary()

# model.fit trains the model
# The validation_split param tells Keras what % of our training data should be used in the validation set
# You can see the validation loss decreasing slowly when you run this
# Because val_loss is no longer decreasing we stop training to prevent overfitting
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# Evaluate the accuracy of our trained model
score = model.evaluate(x_validate, y_validate,
                       batch_size=batch_size, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# print(score)

# Here's how to generate a prediction on individual examples

# for i in range(10):
#     prediction = model.predict(np.array([x_validate[i]]))
#     predicted_label = text_labels[np.argmax(prediction)]
#     print(validate_texts.iloc[i][:50], "...")
#     print('Actual label:' + validate_tags.iloc[i])
#     print("Predicted label: " + predicted_label + "\n")


def gen_filename():
    return str(model_no)+'_'+str(epochs)+'_'+str(max_words)+'_'+\
           str(history.history['val_acc'][-1]).replace('.', ',')[:5]


# save model
model.save('model_'+gen_filename()+'.h5')  # creates a HDF5 file 'my_model.h5'


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
    indexes = []
    results = []

    for i, row in testData.iterrows():
        prediction = model.predict(np.array([x_test[i]]))
        print(prediction)
        #label_id = all_subcategories[predicted_label]
        #indexes.append(row["itemid"])
        #results.append(label_id)

    df = pd.DataFrame({'itemid': indexes, 'Category': results})
    df.to_csv(path_or_buf='res'+gen_filename()+'.csv', index=False)


if gen_test:
    perform_test()

# This utility function is from the sklearn docs: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
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
def plotting():
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
    plt.figure(figsize=(24,20))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    plt.show()

