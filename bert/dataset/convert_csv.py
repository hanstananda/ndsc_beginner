from sklearn.model_selection import train_test_split
import pandas as pd
import csv
from sklearn.utils import shuffle

trainData = pd.read_csv('data/new_train.csv')
testData = pd.read_csv('data/new_test.csv')

train_data_fashion = trainData[trainData['image_path'].str.contains("fashion")]
train_data_beauty = trainData[trainData['image_path'].str.contains("beauty")]
train_data_mobile = trainData[trainData['image_path'].str.contains("mobile")]
test_data_fashion = testData[testData['image_path'].str.contains("fashion")]
test_data_beauty = testData[testData['image_path'].str.contains("beauty")]
test_data_mobile = testData[testData['image_path'].str.contains("mobile")]

# Shuffle train data
train_data_fashion = shuffle(train_data_fashion)
train_data_beauty = shuffle(train_data_beauty)
train_data_mobile = shuffle(train_data_mobile)

train_data_fashion, eval_data_fashion = train_test_split(train_data_fashion, test_size=0.1)
train_data_beauty, eval_data_beauty = train_test_split(train_data_beauty, test_size=0.1)
train_data_mobile, eval_data_mobile = train_test_split(train_data_mobile, test_size=0.1)

def convert_csv(df, filename, category):
    new_rows = []
    for index, row in df.iterrows():
        new_row = ['0' for i in range(3)]
        new_row[0] = row['itemid']
        new_row[1] = row['title']
        if 'test' not in filename:
            if category == 'beauty':
                new_row[2] = str(row['Category'])
            elif category == 'fashion':
                new_row[2] = str(row['Category'] - 17)
            else:
                new_row[2] = str(row['Category'] - 31)
        new_rows.append(new_row)
    with open('data/' + filename + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

convert_csv(train_data_mobile, 'train_data_mobile', 'mobile')
convert_csv(eval_data_mobile, 'eval_data_mobile', 'mobile')
convert_csv(test_data_mobile, 'test_data_mobile', 'mobile')

convert_csv(train_data_beauty, 'train_data_beauty', 'beauty')
convert_csv(eval_data_beauty, 'eval_data_beauty', 'beauty')
convert_csv(test_data_beauty, 'test_data_beauty', 'beauty')

convert_csv(train_data_fashion, 'train_data_fashion', 'fashion')
convert_csv(eval_data_fashion, 'eval_data_fashion', 'fashion')
convert_csv(test_data_fashion, 'test_data_fashion', 'fashion')
