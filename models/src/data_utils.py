import torch
import csv
from torch.utils.data import Dataset
import numpy as np
from collections import Counter
from models.src.pre_processing_for_train import Tokenizer, hformat
from copy import copy
import sys
from bisect import bisect_right
        

def majority_vote(list_of_text_options, ignore_empty=True, if_draw='OUT'):
    if ignore_empty:
        list_of_text_options = [t for t in list_of_text_options if t != None and t != '']
    if len(list_of_text_options) == 0:
        return if_draw
    
    cnt = Counter(list_of_text_options)
    first, cnt_first = cnt.most_common()[0]
    if len(cnt) > 1:
        second, cnt_second = cnt.most_common()[1]
        if cnt_first == cnt_second:
            first = if_draw
    return first

def collate_tensor_data(data_list, binary=False): 
    # TODO: decide binary or multiclass by a parameter

    # Assume a list of pairs (tensor, target) as input
    # Sort tensors by length and then stack them
    data_list.sort(key=lambda b: len(b[0]), reverse=True)
    X_seq, Y_seq = zip(*data_list)
    lengths = [len(X) for X in X_seq]
    X = torch.nn.utils.rnn.pad_sequence(X_seq)
    
    if binary: # binary
        Y = torch.FloatTensor(Y_seq).view(-1,1)
    else: #multiclass
        Y = torch.LongTensor(Y_seq)
    return ((X, lengths), Y)

class TaggedDataset(Dataset):
    def __init__(self, csv_data_file, tags=None, text_field='text', tag_prefix='tag', 
                 vocabulary=None, remove_duplicates=False, make_proportions_equal=False,
                 oversampling={}, min_freq_voc=1, unk_symbol='<unk>', 
                 remove_names=False, remove_accents=False, remove_contiguous_chars=False, 
                 min_examples_per_class=1, verbose=False, to_report=10000, **name_finder_kwargs):
        '''
        Class for reading a csv file with tagged data into a torch Dataset. 
        The csv file is expected to have a field of name 'text_field' and at least one other field
        which name starts with 'tag_prefix'. All fields that start with 'tag_prefix' are considered
        and a majority vote decides the real tag.

        The parameter 'tags' defines the name of the tags considered. If 'tags' is None, 
        then all tags (useful for multiclass classifications in which there are too many classes). 
        in the file are considered. If len(tags)==1 we assume a binary classification with all 
        tags different to the one provided as the negative class.

        The class also contains functionalities to preprocess the data with parameters
        'remove_names', 'remove_accents', 'remove_contiguous_chars'.

        All other parameters should be self explanatory :-)
        '''
        
        self.tokenizer = Tokenizer(voc=None, remove_names=remove_names, 
                              remove_accents=remove_accents, remove_contiguous_chars=remove_contiguous_chars,
                              **name_finder_kwargs)

        self.unk_symbol = unk_symbol
        self.target_mapping = {}
        self._single_tag = False
        self._all_tags = False

        self.tags = copy(tags)
        if self.tags == None:
            self._all_tags = True
            self.tags = set()
        
        if len(self.tags) == 1:
            self._single_tag = True
            tag_text = self.tags[0]
            neg_tag_text = 'NO_' + tag_text
            self.tags.append(neg_tag_text)
                
        self.corpus = ''
        self.text_data = []
        self.verbose = verbose
        
        if self.verbose:
            print('Reading input file...')
        with open(csv_data_file) as infile:
            reader = csv.DictReader(infile)
            tag_fields = [field for field in reader.fieldnames if field.startswith(tag_prefix)]
            for row in reader:
                text = row[text_field]
                tags = [row[field] for field in tag_fields]
                tag_text = majority_vote(tags)
                if tag_text in self.tags:
                    self.text_data.append((text, tag_text))
                elif self._single_tag:
                    self.text_data.append((text, neg_tag_text))
                elif self._all_tags:
                    self.tags.add(tag_text)
                    self.text_data.append((text, tag_text))

        if remove_duplicates:
            if self.verbose:
                print('Removing duplicates...')
            self.text_data = list(set(self.text_data))

        if min_examples_per_class > 1:
            _temp_data = []
            self.tags = set()
            cnt_tags = Counter([tag for text, tag in self.text_data])
            for i,d in enumerate(self.text_data):
                text, tag_text = d
                if cnt_tags[tag_text] > min_examples_per_class:
                    _temp_data.append((text, tag_text)) 
                    self.tags.add(tag_text)
            self.tags = list(self.tags)
            self.text_data = _temp_data
 
        if self._all_tags:
            self.tags = list(self.tags)

        for i,tag in enumerate(self.tags):
            self.target_mapping[tag] = i



        self.corpus = ' '.join([text for (text, tag_text) in self.text_data])
        if vocabulary == None:
            if self.verbose:
                print('Generating vocabulary.')
            self.pre_vocabulary = self.tokenizer.generate_voc(self.corpus, min_freq_voc, self.verbose)
            self.vocabulary = {}
            self.vocabulary[self.unk_symbol] = 0
            for i, (word, cnt) in enumerate(self.pre_vocabulary.most_common()):
                self.vocabulary[word] = i + 1                       
        else:
            if type(vocabulary) == str:
                # TODO: read vocabulary from file
                pass
            else:
                self.vocabulary = vocabulary

        self.idx_to_word = {v:k for k,v in self.vocabulary.items()}
        
        if make_proportions_equal:
            oversampling, count = {}, {}
            max_count_tag = 0
            for tag_text in self.tags:
                count[tag_text] = len([1 for (text,t) in self.text_data if t == tag_text])
                if count[tag_text] > max_count_tag:
                    max_count_tag = count[tag_text]
            for tag_text in self.tags:
                if count[tag_text] != 0:
                    oversampling[tag_text] = max_count_tag/count[tag_text]
                else:
                    oversampling[tag_text] = 0
          
        if self.verbose:
            print('Generating labelled examples...')

        if oversampling != {}:
            if self.verbose:
                print('Oversampling with proportions:', oversampling)
            # we need to oversample data             
            self.data = []
            for i,d in enumerate(self.text_data):
                text, tag_text = d
                if self.verbose and (i % to_report) == (to_report - 1):
                    partial_info = f'\rOversampling labeled examples: {hformat(i)}/{hformat(len(self.text_data))}       '
                    sys.stdout.write(partial_info)
                if tag_text in oversampling:
                    total = oversampling[tag_text]
                else:
                    total = 1
                integer = int(total)
                fraction = total - integer
                if integer >= 1:
                    self.data += [(text, tag_text)] * integer
                if np.random.rand() <= fraction:
                    self.data.append((text, tag_text))
            del self.text_data
        else:
            self.data = self.text_data

    def tensor_to_text(self, tensor_data):
        tokens = [self.idx_to_word[int(idx)] for idx in tensor_data]
        return ' '.join(tokens)
    
    def create_batch_from_raw_data(self, text_list):
        data = []
        for text in text_list:
            tokens = self.tokenizer.tokenize_with_voc(text, self.vocabulary, unk=self.unk_symbol)
            tensor = self.tokens_to_tensor(tokens)
            data.append((tensor,0)) # fake class to simplify
        ((X, lengths), Y) = self.collate_data(data)
        return (X, lengths)

    def tokens_to_tensor(self, tokens):
        x_indexes = [self.vocabulary[word] for word in tokens]
        x_tensor = torch.tensor(x_indexes)
        return x_tensor
        
    def __len__(self):
        return len(self.data)
      
    def __getitem__(self, idx):
        text, tag_text = self.data[idx]
        cl = self.target_mapping[tag_text]   
        tokens = self.tokenizer.tokenize_with_voc(text, self.vocabulary, unk=self.unk_symbol)
        tensor_output = (self.tokens_to_tensor(tokens),cl)
        return tensor_output

    def collate_data(self, data_list, binary=False):
        # TODO: decide if binary or multiclass by a parameter
        # Assume a list of pairs (tensor, target) as input
        # Sort tensors by length and then stack them
        data_list.sort(key=lambda b: len(b[0]), reverse=True)
        X_seq, Y_seq = zip(*data_list)
        lengths = [len(X) for X in X_seq]
        X = torch.nn.utils.rnn.pad_sequence(X_seq)
        
        if binary: 
            Y = torch.FloatTensor(Y_seq).view(-1,1)
        else: #multiclass
            Y = torch.LongTensor(Y_seq)
        return ((X, lengths), Y)


class IntToOneHotVectorConverter():
    '''
    Convierte un entero a un one-hot vector dependiendo de valores tresholds.
    '''
    def __init__(self, limits):
        self._limits = sorted(limits)
        self._dim = len(self._limits) + 1

    def to_index(self, value):
        index = bisect_right(self._limits, value)
        return index

    def list_to_index_array(self, l):
        out_list = [self.to_index(x) for x in l]
        out = np.array(out_list, dtype=np.int32)
        return out

    def list_to_index_tensor(self, l):
        out_array = self.list_to_index_array(l)
        out = torch.from_numpy(out_array)
        return out

    def tensor_to_index_tensor(self, t):
        input_list = [x.item() for x in t]
        out = self.list_to_index_tensor(input_list)
        return out

    def to_one_hot_array(self, value):
        out = np.zeros(self._dim, dtype=np.int32)
        out[index] = True
        return out

    def to_one_hot_tensor(self, value):
        v = self.to_one_hot_array(value)
        out = torch.from_numpy(v)
        return out

    def list_to_one_hot_tensor(self, l, dtype=torch.float32):
        out_list = []
        for val in t:
            v = self.to_one_hot_tensor(val)
            out_list.append(v)
        out = torch.stack(out_list)
        return(out)

    def tensor_to_one_hot_tensor(self, t, dtype=torch.float32):
        input_list = [x.item() for x in t]
        out = self.list_to_one_hot_tensor(input_list)
        return out