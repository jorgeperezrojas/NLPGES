import codecs
import csv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from TFIDF.utils import open_data, split_proporcionaly

PATH_TO_DATA = "/Users/sebastiandonoso/Documents/NLP-GES/wlCorpus.csv"
all_data = open_data(PATH_TO_DATA)
train_data, test_data = split_proporcionaly(0.75, 0.25, all_data)
arr_test = []
arr_train = []
for i in test_data.values():
    arr_test += i
for i in train_data.values():
    arr_train += i
len_test = len(arr_test)
len_train = len(arr_train)
train_num = []
train_key=[]
for key in train_data.keys():
    train_num.append(len(train_data[key]))
    train_key.append(key)
test_num = []
test_key =[]
for key in test_data.keys():
    test_num.append(len(test_data[key]))
    test_key.append(key)
tfid = TfidfVectorizer(ngram_range=(4,4)).fit_transform(arr_test + arr_train)

wait = len_test

def response_presta(i, related_docs,wait):
    res_test=''
    res_train=''
    for pos, j in enumerate(test_num):
        if  i<sum(test_num[:pos+1]):
            res_test = test_key[pos]
    for pos, j in enumerate(train_num):
        if related_docs[0]<sum(train_num[:pos+1]):
            res_train = train_key[pos]
    print(res_test + " | | " + res_train )
    wait -= 1



Q = tfid[0:len_test]
R = tfid[len_test:]

for i in range(len_test):
    cosine_similarites = cosine_similarity(Q[i:i + 1], R).flatten()
    related_docs = cosine_similarites.argsort()[:-5:-1]
    response_presta(i, related_docs,wait)

"""

file1 = codecs.open('MyFile.txt', 'w', 'utf8')
for data in all_data.keys():
    file1.write(str(data) + "  " + str(len(all_data[data])) + '\n')
file1.close()

"""