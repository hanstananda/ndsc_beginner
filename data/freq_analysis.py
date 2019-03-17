import pandas as pd
import csv
import json
from operator import itemgetter

# maps the numbers to subcategories
with open('data/categories.json') as f:
    cats = json.load(f)
subcategories_dic = {}
categories_dic = {}
for category in cats.keys():
    for subcat, index in cats[category].items():
        categories_dic[index] = category
        subcategories_dic[index] = subcat

# count the number of occurrences of all words
df_word_count = pd.read_csv('data/word_freq.csv')
word_list = df_word_count['word'].tolist()
count_list = df_word_count['count'].tolist()
dic_word_count = {}
for i in range(len(df_word_count)):
    dic_word_count[word_list[i]] = count_list[i]

train_data = pd.read_csv('data/clean_train.csv')
df_title_cat = pd.DataFrame(train_data[['title', 'Category']])

# count the number of occurrences of all subcategories
subcategory_count = {}
for index, row in df_title_cat.iterrows():
    if row['Category'] not in subcategory_count:
        subcategory_count[row['Category']] = 1
    else:
        subcategory_count[row['Category']] += 1

def format_output(dic):
    sorted_list = []
    for key, value in dic.items():
        temp = []
        freq = value / subcategory_count[key[2]]
        subcategory = subcategories_dic[key[2]]
        category = categories_dic[key[2]]
        temp.extend(key[:2])
        temp.append(subcategory)
        temp.append(category)
        temp.append(value)
        temp.append(freq)
        sorted_list.append(temp)
    sorted_list = sorted(sorted_list, key=lambda x: (x[1], x[5]), reverse=True)
    return sorted_list

word_dic = {}
for index, row in df_title_cat.iterrows():
    for word in row['title'].split():
        # 'nan' as a keyword gives an error
        if word != 'nan':
            if (word, dic_word_count[word], row['Category']) not in word_dic:
                word_dic[(word, dic_word_count[word], row['Category'])] = 1
            else:
                word_dic[(word, dic_word_count[word], row['Category'])] += 1
res_list = format_output(word_dic)

header = ['Word', 'Total Occurences', 'Subcategory', 'Category', 'Occurences', '% in Subcategory']
with open('data/word_cat_freq.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow([i for i in header])
    writer.writerows(res_list)