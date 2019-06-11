import json
from io import open

import json



"""

{
    {
        'general': '<---->',
        'specific': [
            'pathology', ....., 'pathology'
        ],
        'age_range': '<---->',
        'fields': ['<----->', '<----->', '<----->']
    }
        .
        .
        .
    {
        'general': '<---->',
        'specific': [
            'pathology', ....., 'pathology'
        ],
        'age_range': '<---->',
        'fields': ['<----->', '<----->', '<----->']
    }
}
"""
# open file of ges pathology

with open('../data/ges-health-problems.txt', encoding="utf8") as json_file:
    data = json.load(json_file)

# example json structcture




def create_specific(specific):
    new_array = list()
    for j in specific:
        new_array.append(j)
    return new_array


def create_json_object(general, specific):
    new_dict = dict()
    new_dict['general'] = general
    new_dict['specific'] = create_specific(specific)
    new_dict['fields'] = []
    new_dict['all_ranges'] = []
    return new_dict


# create json file
new_json = list()
for i in data:
    new_json.append(create_json_object(i, data[i]))

# obtain
import pandas as pd

file_errors_location = '../data/ges-health-problems-labeled.xlsx'
df = pd.read_excel(file_errors_location)



# add speciality field to json label ges

for index, row in df.iterrows():
    for j in row[1:4]:
        if type(j) is str:
            new_json[index]['fields'].append(j)

# read csv of specialities range ages
ages_data = pd.read_excel("../data/ges-health-problems.xlsx")


def add_range_json(json, min_age, max_age, pathology):
    for i in json:
        for j in i['specific']:
            if pathology == j:
                if min_age not in i['all_ranges']:
                    i['all_ranges'].append(min_age)
                if max_age not in i['all_ranges']:
                    i['all_ranges'].append(max_age)
                i['all_ranges'].sort()

                return


# add range age to labeled ges
# la edad es igual para todos los patology
for index, row in ages_data.iterrows():
    add_range_json(new_json, row[1], row[2], row[0])

with open('../data/some_file.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(new_json, ensure_ascii=False))
