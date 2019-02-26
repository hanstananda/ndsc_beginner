import os

import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow.keras.backend as K

from tensorflow.keras import layers

data_root = "../../beauty_image/"

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"


def feature_extractor(x):
    feature_extractor_module = hub.Module(feature_extractor_url)
    return feature_extractor_module(x)


IMAGE_SIZE = hub.get_expected_image_size(hub.Module(feature_extractor_url))
image_data = image_generator.flow_from_directory(os.path.join(data_root, "Train"), target_size=IMAGE_SIZE)
for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break

features_extractor_layer = layers.Lambda(feature_extractor, input_shape=IMAGE_SIZE+[3])
features_extractor_layer.trainable = False
model = tf.keras.Sequential([
    features_extractor_layer,
    layers.Dense(image_data.num_classes, activation='softmax')
])
model.summary()

sess = K.get_session()
init = tf.global_variables_initializer()
sess.run(init)
result = model.predict(image_batch)

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss='categorical_crossentropy',
    metrics=['accuracy'])

class CollectBatchStats(tf.keras.callbacks.Callback):
    def __init__(self):
        self.batch_losses = []
        self.batch_acc = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs['loss'])
        self.batch_acc.append(logs['acc'])


steps_per_epoch = image_data.samples//image_data.batch_size
batch_stats = CollectBatchStats()
model.fit((item for item in image_data), epochs=50,
          steps_per_epoch=steps_per_epoch,
          callbacks = [batch_stats])

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats.batch_losses)

plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats.batch_acc)
plt.show()

label_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
label_names = np.array([key.title() for key, value in label_names])

result_batch = model.predict(image_batch)
labels_batch = label_names[np.argmax(result_batch, axis=-1)]

# plt.figure(figsize=(10,9))
# for n in range(30):
#     plt.subplot(6,5,n+1)
#     plt.imshow(image_batch[n])
#     plt.title(labels_batch[n])
#     plt.axis('off')
#     _ = plt.suptitle("Model predictions")
#     plt.show()

export_path = tf.contrib.saved_model.save_keras_model(model, "../checkpoints/")
export_path