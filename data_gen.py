from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

TRAIN_DIR = "../mobile_image"
HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=90,
      horizontal_flip=True,
      vertical_flip=True
    )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE)
