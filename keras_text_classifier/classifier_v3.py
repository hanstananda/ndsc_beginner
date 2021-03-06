# This is used for testing Fine Tune Hyper-Parameters

from datetime import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras_preprocessing.sequence import pad_sequences
import joblib
from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, LSTM
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
max_length = 35  # 32 is max word in train, think need to test this later of the actual words need to be used...
num_classes = len(all_subcategories)
# Training for more epochs will likely lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 10
max_words = 1000

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

trainData = pd.read_csv("../data/train.csv")

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

# Pad sequences with zeros
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

y_train = train_tags.values
y_train = utils.to_categorical(y_train)
vocab_size = len(tokenize.word_index) + 1
print(vocab_size)


def create_model(num_filters, kernel_size, max_words, embedding_dim, max_length):
    model = Sequential()
    model.add(Embedding(max_words,
                        embedding_dim,
                        input_length=max_length,
                        trainable=True))
    model.add(Conv1D(num_filters, kernel_size, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(embedding_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def gen_filename_h5():
    return 'epoch_'+str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


def gen_filename_csv():
    return 'epoch_'+str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


param_grid = dict(num_filters=[32, 64, 128],
                  kernel_size=[3, 5, 7],
                  max_words=[max_words],
                  embedding_dim=[64, 128],
                  max_length=[max_length])

filepath = "../checkpoints/"+gen_filename_h5()+"v2.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
model = KerasClassifier(build_fn=create_model,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=True,
                        )

grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                              cv=4, verbose=1, n_iter=10)

print(grid)

grid_result = grid.fit(x_train,
                       y_train,
                       validation_split=0.1,
                       callbacks=[checkpointer])


with open("../checkpoints/"+gen_filename_h5()+".pickle","w+") as f:
    joblib.dump(grid_result, f)


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
    plot_history(grid)


def perform_test():
    prediction = grid.predict(x_test, batch_size=batch_size, verbose=1)
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
