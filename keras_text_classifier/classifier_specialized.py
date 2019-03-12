from datetime import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text
from keras import utils


class Classifier:
    """
    Classifier class
    """
    test_data_path = '../data/test.csv'

    dict_data_path = '../data/kata_dasar_kbbi.csv'

    categories_path = '../data/categories.json'

    train_data_path_with_cname = '../data/train_with_cname.csv'

    train_data_path = '../data/train.csv'

    categories = ['Fashion', 'Beauty', 'Mobile']

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

    def __init__(self, max_words=2500, plot_history_check=True, gen_test=256, batch_size=256, epoch=30):

        self.max_words = max_words

        self.plot_history_check = plot_history_check

        self.gen_test = gen_test

        self.batch_size = batch_size

        self.epoch = epoch

        self.test_data = None

        self.dict_data = None

        self.train_data = None

        self.categories = None

        self.train_data = None

        self.tokenizer = None

        self.encoder = LabelEncoder()

        self.load_test_data()
        self.load_dic_data()
        self.load_categories()

    def load_test_data(self):
        self.test_data = pd.read_csv(self.test_data_path)

    def load_dic_data(self):
        self.dict_data = pd.read_csv(self.dict_data_path)

    def load_categories(self):
        o = open(self.categories_path, "r")
        self.categories = json.load(o)

    def load_train_data(self):
        try:
            self.train_data = pd.read_csv(self.train_data_path_with_cname)
            print("custom train data used")
        except:
            print("cannot find custom data, generating...")
            self.train_data = pd.read_csv(self.train_data_path)
            self.train_data['item_category'] = 'None'
            for index, row in self.train_data.iterrows():
                s = row["title"]
                img_path = row["image_path"]
                cat = self.category_mapping[img_path.split('/')[0]]
                if cat == 'Fashion':
                    sub_cats = self.get_inverted_subcategory('Fashion')
                elif cat == 'Mobile':
                    sub_cats = self.get_inverted_subcategory('Mobile')
                elif cat == 'Beauty':
                    sub_cats = self.get_inverted_subcategory('Beauty')
                # trainData.set_value(index, 'item_category', sub_cats[row['Category']])
                self.train_data.at[index, 'item_category'] = sub_cats[row['Category']]
            try:
                self.train_data.to_csv(path_or_buf=self.train_data_path_with_cname, index=False)
            except:
                self.train_data.to_csv(path_or_buf='train_with_cname.csv', index=False)

    def get_all_subcategories(self):
        all_subcategories = {}
        for category in self.categories:
            subcategory = self.get_subcategory(category)
            all_subcategories.update(subcategory)
        return all_subcategories

    def get_inverted_subcategory(self, category):
        return {v: k.lower() for k, v in self.categories[category.capitalize()].items()}

    def get_subcategory(self, category):
        return {k.lower(): v for k, v in self.categories[category].items()}

    def get_train_data(self, category, test=False):
        if test:
            self.load_test_data()
            return self.test_data[self.test_data['image_path'].str.contains(category)]
        else:
            print('get_train_data')
            self.load_train_data()
            return self.train_data[self.train_data['image_path'].str.contains(category)]

    def shuffle_train_data(self, category, test=False):
        return shuffle(self.get_train_data(category, test))

    def get_train_text(self, category, test=False):
        print('shuffle')
        return self.get_train_data(category, test)['title']

    def get_train_tag(self, category, test=False):
        print('get train tag')
        return self.shuffle_train_data(category, test)['item_category']

    def tokenize(self, category, test=False):
        self.tokenizer = text.Tokenizer(num_words=self.max_words, char_level=False)
        self.tokenizer.fit_on_texts(self.get_train_text(category, test))

    def text_to_matrix(self, category, test=False):
        self.tokenize(category, test)
        return self.tokenizer.texts_to_matrix(self.get_train_text(category, test))

    def fit_encoder(self, category, test=False):
        return self.encoder.fit(self.get_train_tag(category, test))

    def transform_encoder(self, category, test=False):
        self.fit_encoder(category, test)
        return self.encoder.transform(self.get_train_tag(category, test))

    def one_hot_repr(self, category, test=False):
        return utils.to_categorical(self.transform_encoder(category, test), len(self.get_inverted_subcategory(category)))

    def generate_model(self, category):
        model = Sequential()
        model.add(Dense(512, input_shape=(self.max_words,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.get_inverted_subcategory(category))))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model

    @classmethod
    def build_model(cls, category):
        cls.generate_model(category)

    def gen_filename_h5(self, history):
        return 'epoch_' + str(self.epoch) + '_' + str(self.max_words) + '_' + \
               str(history.history['val_acc'][-1]).replace('.', ',')[:5]

    def gen_filename_csv(self):
        return 'epoch_' + str(self.epoch) + '_' + str(self.max_words) + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    @staticmethod
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

    def train(self, category, test=False):
        model = self.generate_model(category)
        return model.fit(
            self.text_to_matrix(category, test),
            self.one_hot_repr(category, test),
            batch_size=self.batch_size,
            epochs=self.epoch,
            verbose=1,
            validation_split=0.1
        )
        # if self.plot_history_check:
        #     self.plot_history(model)

        # return model

    def save(self, category, test=False):
        name = 'model_' + str(category) + '_' + self.gen_filename_h5(self.train(category, test)) + '.h5'
        return self.generate_model(category).save(name)

    def perform_test(self, category):
        model = self.save(category, False)
        prediction = model.predict(self.text_to_matrix(category, test=True), self.batch_size, verbose=1)
        all_subcategories = self.get_all_subcategories()
        predicted_label = [all_subcategories[self.fit_encoder(category, False).classes_[np.argmax(prediction[i])]]
                           for i in range(len(self.text_to_matrix(category, test=True)))
                           ]
        return predicted_label

    @classmethod
    def data_frame(cls, category):
        return pd.DataFrame({'itemid': cls.get_train_data(category)['itemid'].astype(int), 'Category': cls.perform_test(category)})
        # print(predicted_label_fashion)
        # print(prediction_beauty)
        # print(prediction_mobile)

    @classmethod
    def plot_confusion_matrix(cls, cm, classes,
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
    @classmethod
    def plotting(cls, model, text_labels, x_validate, y_validate):
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
        cls.plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
        plt.show()


classifier = Classifier()

test_data_fashion = classifier.get_train_data('fashion', True)['itemid'].astype(int)
test_data_beauty = classifier.get_train_data('beauty', True)['itemid'].astype(int)
test_data_mobile = classifier.get_train_data('mobile', True)['itemid'].astype(int)

predicted_label_fashion = classifier.perform_test('Fashion')
predicted_label_beauty = classifier.perform_test('Beauty')
predicted_label_mobile = classifier.perform_test('Mobile')


df = pd.DataFrame({'itemid': test_data_fashion, 'Category': predicted_label_fashion})
df = df.append(pd.DataFrame({'itemid': test_data_beauty, 'Category': predicted_label_beauty}))
df = df.append(pd.DataFrame({'itemid': test_data_mobile, 'Category': predicted_label_mobile}))

df.to_csv('res' + classifier.gen_filename_csv() + '.csv', index=False)
