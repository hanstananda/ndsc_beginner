import os

import tensorflow as tf
import tensorflow_hub as hub

from utility.train_data_loader import load_train_data

data_root = "../../fashion_image/"

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"

trainData = load_train_data()
train_data_specialized = trainData[trainData['image_path'].str.contains("fashion")]
train_data_specialized['image_path'] = train_data_specialized['image_path'].map(lambda x: x.lstrip('fashion_image/'))
# print(train_data_specialized)
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
image_data = image_generator.flow_from_dataframe(train_data_specialized,
                                                 directory=os.path.join(data_root),
                                                 x_col="image_path",
                                                 y_col="item_category",
                                                 target_size=IMAGE_SIZE,
                                                 color_mode="rgb",
                                                 class_mode="categorical",
                                                 shuffle=True,
                                                 )


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break



