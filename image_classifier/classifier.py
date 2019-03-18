from datetime import datetime
import json
import os
import numpy as np

import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import matplotlib.pyplot as plt

from utility.train_data_loader import load_train_data

epochs = 10
batch_size = 256
specialization = "beauty"
gen_test = False

categories_file = open("../data/categories.json", "r")
categories = json.load(categories_file)

all_subcategories = {k.lower(): v for k, v in categories['Mobile'].items()}
all_subcategories.update({k.lower(): v for k, v in categories['Fashion'].items()})
all_subcategories.update({k.lower(): v for k, v in categories['Beauty'].items()})

data_root = f"../../{specialization}_image/"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

trainData = load_train_data()
testData = pd.read_csv("../data/test.csv")

# train_data_specialized = trainData[trainData['image_path'].str.contains(specialization)][::]
# train_data_specialized['image_path'] = train_data_specialized['image_path']. \
#     map(lambda x: x.replace(f"{specialization}_image/", ''))
#
# validation_data_specialized = trainData[trainData['image_path'].str.contains(specialization)][::10]
# validation_data_specialized['image_path'] = validation_data_specialized['image_path']. \
#     map(lambda x: x.replace(f"{specialization}_image/", ''))
#

test_data_specialized = testData[testData['image_path'].str.contains(specialization)]
test_data_specialized['image_path'] = test_data_specialized['image_path']. \
    map(lambda x: x.replace(f"{specialization}_image/", ''))

inverted_categories_specialized = {k.lower(): v for k, v in categories[specialization.capitalize()].items()}

train_data_specialized = trainData[trainData['image_path'].str.contains(specialization)][::]
df_train = pd.DataFrame()
df_valid = pd.DataFrame()
num_train = 2000
num_valid = int(0.1 * num_train)
for k, v in inverted_categories_specialized.items():
    rows = train_data_specialized.loc[train_data_specialized['Category'] == v]
    num_images = rows.shape[0]
    if num_train + num_valid > num_images:
        nt = int(0.9 * num_images)
        nv = int(0.1 * num_images)
    else:
        nt = num_train
        nv = num_valid
    # print(nt,nv)
    rows_train = rows[:nt]
    df_train = df_train.append(rows_train)
    rows_valid = rows[nt:(nt + num_valid)]
    df_valid = df_valid.append(rows_valid)

train_data_specialized = df_train
validation_data_specialized = df_valid

train_data_specialized['image_path'] = train_data_specialized['image_path']. \
    map(lambda x: x.replace(specialization + '_image/', ''))

validation_data_specialized['image_path'] = validation_data_specialized['image_path']. \
    map(lambda x: x.replace(specialization + '_image/', ''))


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
image_generator = datagen.flow_from_dataframe(train_data_specialized,
                                              directory=os.path.join(data_root),
                                              x_col="image_path",
                                              y_col="item_category",
                                              target_size=IMAGE_SIZE,
                                              color_mode="rgb",
                                              class_mode="categorical",
                                              shuffle=True,
                                              batch_size=64,
                                              )

valid_generator = valid_datagen.flow_from_dataframe(validation_data_specialized,
                                                    directory=os.path.join(data_root),
                                                    x_col="image_path",
                                                    y_col="item_category",
                                                    target_size=IMAGE_SIZE,
                                                    color_mode="rgb",
                                                    class_mode="categorical",
                                                    shuffle=True,
                                                    batch_size=64,
                                                    )

test_generator = test_datagen.flow_from_dataframe(test_data_specialized,
                                                  directory=os.path.join(data_root),
                                                  x_col="image_path",
                                                  y_col=None,
                                                  target_size=IMAGE_SIZE,
                                                  color_mode="rgb",
                                                  class_mode=None,
                                                  shuffle=False,
                                                  batch_size=64,
                                                  )

label_names = sorted(image_generator.class_indices.items(), key=lambda pair: pair[1])
label_names = np.array([key.title() for key, value in label_names])


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


for image_batch, label_batch in image_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

input_shape = IMAGE_SIZE+[3]

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(inverted_categories_specialized), activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


def gen_filename_h5():
    return 'epoch_' + str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


def gen_filename_csv():
    return 'epoch_' + str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# Checkpoint auto
filepath = f"../checkpoints/{gen_filename_h5()}v2.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

steps_per_epoch = image_generator.samples // image_generator.batch_size
valid_steps_per_epoch = valid_generator.samples // valid_generator.batch_size
test_steps_per_epoch = test_generator.samples // test_generator.batch_size

history = model.fit_generator(generator=image_generator,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=valid_generator,
                              validation_steps=valid_steps_per_epoch,
                              epochs=epochs,
                              callbacks=[checkpointer],
                              )

model.save(f"model_{specialization}_image{gen_filename_h5()}.h5")


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


plot_history(history)


def perform_test():
    prediction_specialized = model.predict_generator(test_generator, verbose=1, steps=test_steps_per_epoch)
    predicted_label_specialized = [np.argmax(prediction_specialized[i]) for i in range(len(prediction_specialized))]
    df = pd.DataFrame({'itemid': test_data_specialized['itemid'].astype(int), 'Category': predicted_label_specialized})
    df.to_csv(path_or_buf='res' + gen_filename_csv() + '.csv', index=False)


if gen_test:
    perform_test()
