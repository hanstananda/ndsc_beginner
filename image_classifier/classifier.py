from datetime import datetime
import json
import os
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Dense

from utility.train_data_loader import load_train_data

epochs = 10
specialization = "beauty"

categories_file = open("../data/categories.json", "r")
categories = json.load(categories_file)

data_root = "../../"+specialization+"_image/"

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

trainData = load_train_data()
train_data_specialized = trainData[trainData['image_path'].str.contains(specialization)][::10]
train_data_specialized['image_path'] = train_data_specialized['image_path'].\
    map(lambda x: x.lstrip(specialization+'_image/'))

validation_data_specialized = trainData[trainData['image_path'].str.contains(specialization)][::100]
validation_data_specialized['image_path'] = validation_data_specialized['image_path'].\
    map(lambda x: x.lstrip(specialization+'_image/'))

categories_specialized = {k.lower(): v for k, v in categories[specialization.capitalize()].items()}
# print(train_data_specialized)
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

label_names = sorted(image_generator.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


for image_batch, label_batch in image_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break


model = Sequential()
model.add(Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3], trainable=True))
model.add(Dense(len(categories_specialized), activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


def gen_filename_h5():
    return 'epoch_'+str(epochs) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# Checkpoint auto
filepath = "../checkpoints/"+gen_filename_h5()+"v2.hdf5"
checkpointer = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


steps_per_epoch = image_generator.samples//image_generator.batch_size
valid_steps_per_epoch = valid_generator.samples // valid_generator.batch_size

model.fit_generator(generator=image_generator,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=valid_generator,
                    validation_steps=valid_steps_per_epoch,
                    epochs=epochs,
                    callbacks=[checkpointer],
                    )

