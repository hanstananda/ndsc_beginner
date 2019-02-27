import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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

plot_history_check = True
gen_test = False

print(all_subcategories)
print("no of categories: "+str(len(all_subcategories)))

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

try:
    trainData = pd.read_csv("../data/train_with_cname.csv")
    print("custom train data used")
except:
    print("cannot find custom data, generating...")
    trainData = pd.read_csv("../data/train.csv")
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
        trainData.to_csv(path_or_buf='../data/train_with_cname.csv', index=False)
    except:
        trainData.to_csv(path_or_buf='train_with_cname.csv', index=False)


max_data_size = int(len(trainData) * 1)
print(max_data_size)

train_texts = trainData['title']
train_tags = trainData['item_category']
validate_texts = trainData['title'][::max_data_size//100]
validate_tags = trainData['item_category'][::max_data_size//100]
test_texts = testData['title']

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_texts)  # only fit on train
x_train = tokenize.texts_to_matrix(train_texts)
x_validate = tokenize.texts_to_matrix(validate_texts)
x_test = tokenize.texts_to_matrix(test_texts)

# print(x_train[0])
print(len(x_train[0]))

# Use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(train_tags)
y_train = encoder.transform(train_tags)
y_validate = encoder.transform(validate_tags)

# print(y_validate)
print(len(y_validate))

# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
print("num classes:", num_classes)
y_train = utils.to_categorical(y_train, num_classes)
y_validate = utils.to_categorical(y_validate, num_classes)

# Inspect the dimenstions of our training and test data (this is helpful to debug)
print('x_train shape:', x_train.shape)
print('x_validate shape:', x_validate.shape)
print('y_train shape:', y_train.shape)
print('y_validate shape:', y_validate.shape)

# This model trains very quickly and 2 epochs are already more than enough
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 50

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

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
text_labels = encoder.classes_

# for i in range(10):
#     prediction = model.predict(np.array([x_validate[i]]))
#     predicted_label = text_labels[np.argmax(prediction)]
#     print(validate_texts.iloc[i][:50], "...")
#     print('Actual label:' + validate_tags.iloc[i])
#     print("Predicted label: " + predicted_label + "\n")

# save model
model.save('model_epoch_50.h5')  # creates a HDF5 file 'my_model.h5'

def perform_test():
    indexes = []
    results = []

    for i, row in testData.iterrows():
        prediction = model.predict(np.array([x_test[i]]))
        predicted_label = text_labels[np.argmax(prediction)]
        label_id = all_subcategories[predicted_label]
        indexes.append(row["itemid"])
        results.append(label_id)

    df = pd.DataFrame({'itemid': indexes, 'Category': results})
    df.to_csv(path_or_buf='res7.csv', index=False)


if gen_test:
    perform_test()



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

if plot_history_check:
    plot_history(history)

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

