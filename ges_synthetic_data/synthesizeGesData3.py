import csv
import json
import random
import re
from collections import defaultdict


class SynthesizeData:
    def __init__(self, path_text, total=100, fields=False):
        self.total = total
        self.fields = fields
        self.path_text = path_text
        self.pathologies_data = defaultdict(dict)
        self.pathologies_ranges = defaultdict(list)

    def normalice_text(self, text):
        keep = set('aáeéoóíúiuübcdfghjklmnñopqrstvwxyz ')
        text = text.lower()
        text = ''.join([c for c in text if c in keep])
        text = re.sub(' +', ' ', text)
        return text

    def open_data(self):
        with open(self.path_text) as json_file:
            data = json.load(json_file)
            for row in data:
                self.extract_by_pathology(row)

    def extract_by_pathology(self, categorie):
        for specific in categorie['specific']:
            original_text = specific
            key = self.normalice_text(original_text)
            if not self.fields:
                self.pathologies_data[key]['field'] = 'None'
            self.pathologies_data[key]['original_text'] = original_text
            left = int(categorie['all_range'][0])
            max = categorie['all_range'][1]
            right = 100 if max == 'inf' else int(max)
            self.pathologies_ranges[key] = range(left, right)

    def create_synthesize_data(self, out_path, example_for_each_age=True, random_by_pathology=60, random_data=True):
        random.seed(333)
        with open(out_path, 'w') as outfile:
            writer = csv.DictWriter(outfile, ['SOSPECHA_DIAGNOSTICA', 'ESPECIALIDAD', 'EDAD', 'GES'])
            writer.writeheader()
            for key in self.pathologies_data:
                row = dict()
                row['SOSPECHA_DIAGNOSITCA'] = self.pathologies_data[key]['original_text']
                row['ESPECIALIDAD'] = self.pathologies_data[key]['field']
                possible_ages = range(0, 100)
                ges_ages = self.pathologies_ranges[key]
                not_ges_ages = [x for x in possible_ages if x not in ges_ages]
                if example_for_each_age:
                    for i in range(0, 100):
                        row['EDAD'] = i
                        row['GES'] = 'True' if i in ges_ages else 'False'
                        writer.writerow(row)

                if random_data:
                    for i in range(random_by_pathology):
                        if (not_ges_ages == []) or (random.uniform(0, 1) > 0.5):
                            row['GES'] = 'True'
                            row['EDAD'] = random.choice(ges_ages)
                        else:
                            row['GES'] = 'False'
                            row['EDAD'] = random.choice(not_ges_ages)
                        writer.writerow(row)
