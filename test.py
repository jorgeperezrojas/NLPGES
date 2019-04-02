import codecs
import csv

PATH_TO_DATA = "/Users/sebastiandonoso/Documents/NLP-GES/wlCorpus.csv"
all_data = dict()
with codecs.open(PATH_TO_DATA, 'r', 'utf8') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        try:
            all_data[row['PRESTA_EST']].append(row['SOSPECHA_DIAG'])
        except:
            all_data[row['PRESTA_EST']] = [row['SOSPECHA_DIAG']]
file1 = codecs.open('MyFile.txt', 'w', 'utf8')
for data in all_data.keys():
    file1.write(str(data) + "  " + str(len(all_data[data])) + '\n')
file1.close()
