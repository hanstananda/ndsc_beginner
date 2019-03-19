from datetime import datetime
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, Flatten, LSTM, \
    Bidirectional, MaxPooling1D
from keras.preprocessing import text, sequence
from keras import utils
import pandas as pd
import xgboost as xgb


from utility.train_data_loader import load_train_data


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
max_length = 35  # 32 is max word in train
max_words = 2500
num_classes = len(all_subcategories)
# Training for more epochs will likelval-acc after 10 epochs: 0.71306y lead to overfitting on this dataset
# You can try tweaking these hyperparamaters when using this model with your own data
batch_size = 256
epochs = 10

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

trainData = load_train_data()

# Shuffle train data
trainData = shuffle(trainData)

max_data_size = int(len(trainData) * 1)
train_data_size = int(max_data_size * .9)
train_data_step = 1
validate_data_step = 1
print(train_data_size, max_data_size)

train_texts = trainData['title'][:train_data_size:train_data_step]
valid_texts = trainData['title'][train_data_size::train_data_step]
train_tags = trainData['Category'][:train_data_size:train_data_step]
valid_tags = trainData['Category'][train_data_size::train_data_step]
test_texts = testData['title']
print(len(train_texts), len(train_tags))

y = train_tags.values

tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_texts)  # only fit on train
x_train = tokenize.texts_to_sequences(train_texts)
x_valid = tokenize.texts_to_sequences(valid_texts)
x_test = tokenize.texts_to_sequences(test_texts)

word_index = tokenize.word_index

# Pad sequences with zeros
x_train = pad_sequences(x_train, padding='post', maxlen=max_length)
x_valid = pad_sequences(x_valid,padding='post',maxlen=max_length)
x_test = pad_sequences(x_test, padding='post', maxlen=max_length)

y_train = train_tags.values
y_valid = valid_tags.values
vocab_size = len(tokenize.word_index) + 1
print(vocab_size)

dtrain = xgb.DMatrix(x_train,label = y_train)
dvalid = xgb.DMatrix(x_valid,label = y_valid)

# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.2
param['max_depth'] = 6
param['num_class'] = 58
param['verbosity'] = 3
param['tree-method'] = 'gpu-hist'
param['updater'] = 'grow_gpu'

watchlist = [(dtrain, 'train'), (dvalid, 'test')]
num_round = 5000
# bst = xgb.train(param, dtrain, num_round, watchlist,early_stopping_rounds = 50)
res = xgb.cv(param, dtrain, num_round, nfold=3, metrics={'merror'}, callbacks=[xgb.callback.print_evaluation(show_stdv=True)])


pred = bst.predict(dvalid)
error_rate = np.sum(pred != y_valid) / y_valid.shape[0]
print('Test error using softmax = {}'.format(error_rate))
print('Accuracy using softmax = {}'.format(accuracy_score(y_valid,pred)))
bst.save_model('xgboost' + datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

# from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# from sklearn import decomposition, ensemble
#
# import pandas, xgboost, numpy, string
# from keras.preprocessing import text, sequence
# from keras import layers, models, optimizers
#
#
# def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
#     # fit the training dataset on the classifier
#     classifier.fit(feature_vector_train, label)
#
#     # predict the labels on validation dataset
#     predictions = classifier.predict(feature_vector_valid)
#
#     if is_neural_net:
#         predictions = predictions.argmax(axis=-1)
#
#     return metrics.accuracy_score(predictions, valid_y)
#
# # split the dataset into training and validation datasets
# train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainData['title'], trainData['Category'])
#
# # label encode the target variable
# encoder = preprocessing.LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# valid_y = encoder.fit_transform(valid_y)
#
# # create a count vectorizer object
# count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# count_vect.fit(trainData['title'])
# print("")
# # transform the training and validation data using count vectorizer object
# xtrain_count =  count_vect.transform(train_x)
# xvalid_count =  count_vect.transform(valid_x)
#
# # word level tf-idf
# tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
# tfidf_vect.fit(trainData['title'])
# xtrain_tfidf =  tfidf_vect.transform(train_x)
# xvalid_tfidf =  tfidf_vect.transform(valid_x)
#
# # ngram level tf-idf
# tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# tfidf_vect_ngram.fit(trainData['title'])
# xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
# xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
#
# # characters level tf-idf
# tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
# tfidf_vect_ngram_chars.fit(trainData['title'])
# xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
# xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
#
# # Extereme Gradient Boosting on Count Vectors
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
# print ("Xgb, Count Vectors: ", accuracy)
#
# # Extereme Gradient Boosting on Word Level TF IDF Vectors
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
# print ("Xgb, WordLevel TF-IDF: ", accuracy)
#
# # Extereme Gradient Boosting on Character Level TF IDF Vectors
# accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
# print ("Xgb, CharLevel Vectors: ", accuracy)