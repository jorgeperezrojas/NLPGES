import codecs
import csv

from data_preparation_functions import split_proporcionaly

from TFIDF.utils import open_data

PATH_TO_DATA = "/Users/sebastiandonoso/Documents/NLP-GES/wlCorpus.csv"
all_data = open_data(PATH_TO_DATA)
train_data, test_data = split_proporcionaly(0.75, 0.25, all_data)

"""
file1 = codecs.open('MyFile.txt', 'w', 'utf8')
for data in all_data.keys():
    file1.write(str(data) + "  " + str(len(all_data[data])) + '\n')
file1.close()
"""
