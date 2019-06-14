import json
import re


class SynthesizeData:
    def __init__(self, path_text, total=100, fields=False):
        self.total = total
        self.fields = fields
        self.path_text = path_text

    def open_data(self):
        with open(self.path_text) as json_file:
            self.data = json.load(json_file)



    def normalice_text(self, text):
        keep = set('aáeéoóíúiuübcdfghjklmnñopqrstvwxyz ')
        text = text.lower()
        text = ''.join([c for c in text if c in keep])
        text = re.sub(' +', ' ', text)
        return text


