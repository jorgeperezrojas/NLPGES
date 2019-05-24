import torch
import csv
from torch.utils.data import Dataset
from collections import Counter
from models.src.data_utils import TaggedDataset
import numpy as np
from bisect import bisect_right
import ipdb

def collate_ges_tensor_data(data_list, binary=False): 
    # TODO: decide binary or multiclass by a parameter

    # Assume a list of pairs ((tensor_text, especialidad_idx, edad_idx), target) as input
    # Sort tensors by length and then stack them
    data_list.sort(key=lambda b: len(b[0][0]), reverse=True)
    X_seq, Y_seq = zip(*data_list)
    X_text_seq, X_especialidad_seq, X_edad_seq = zip(*X_seq)
    lengths = [len(X) for X in X_text_seq]
    X_text = torch.nn.utils.rnn.pad_sequence(X_text_seq)
    X_especialidad = torch.tensor(X_especialidad_seq)
    X_edad = torch.tensor(X_edad_seq)
    
    if binary: # binary
        Y = torch.FloatTensor(Y_seq).view(-1,1)
    else: #multiclass
        Y = torch.LongTensor(Y_seq)
    return ((X_text, X_especialidad, X_edad, lengths), Y)

def prepare_ges_batch_fn(batch, device):
    (X,Y) = batch
    (X_text, X_especialidad, X_edad, lengths) = X
    X_text = X_text.to(device)
    X_especialidad = X_especialidad.to(device)
    X_edad = X_edad.to(device)
    X = (X_text, X_especialidad, X_edad, lengths)
    Y = Y.to(device)
    return (X,Y)

class GesDataset(Dataset):
    def __init__(self, csv_data_file, text_field='SOSPECHA_DIAGNOSTICA', 
                 ges_field='GES', ges_options=['True','False'], 
                 especialidad_field='ESPECIALIDAD', unk_especialidad='<unk_esp>',
                 edad_field='EDAD', edad_limits=[1,4,5,6,7,9,15,20,25,35,50,55,60,61,65],
                 especialidad_voc=None, verbose=True,
                 **tagged_dataset_kwargs):
        
        self.tagged_dataset = TaggedDataset(csv_data_file, tags=ges_options, 
            text_field=text_field, tag_prefix=ges_field, remove_duplicates=False, 
            make_proportions_equal=False, oversampling={}, verbose=verbose, **tagged_dataset_kwargs)

        self.tags = self.tagged_dataset.tags
        self.vocabulary = self.tagged_dataset.vocabulary

        if verbose:
            print('Reading edad and especialidad info...')

        self.edad_data = []
        self.especialidad_data = []

        with open(csv_data_file) as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                edad = int(float(row[edad_field]))
                self.edad_data.append(edad)
                especialidad = row[especialidad_field]
                self.especialidad_data.append(especialidad)

        assert len(self.tagged_dataset.data) == len(self.edad_data)
        assert len(self.tagged_dataset.data) == len(self.especialidad_data)

        if especialidad_voc != None:
            self.especialidad_voc = especialidad_voc
        else:
            espec_cnt = Counter(self.especialidad_data)
            self._especialidad_indx_to_word = [unk_especialidad] + [x for (x,c) in espec_cnt.most_common()]
            self.especialidad_voc = {w:i for i,w in enumerate(self._especialidad_indx_to_word)}

        self.converter = IntToOneHotVectorConverter(edad_limits)
        self.edad_limits_size = len(edad_limits) + 1
        
    def __len__(self):
        return len(self.tagged_dataset.data)
      
    def __getitem__(self, idx):
        text_tensor, cl = self.tagged_dataset[idx]
        especialidad = self.especialidad_data[idx]
        especialidad_idx = self.especialidad_voc[especialidad]
        edad = self.edad_data[idx]
        edad_idx = self.converter.to_index(edad)
        output = ((text_tensor, especialidad_idx, edad_idx), cl)
        return output

    def get_raw_data(self, idx):
        text, clase = self.tagged_dataset.data[idx]
        especialidad = self.especialidad_data[idx]
        edad = self.edad_data[idx]
        return ((text, especialidad, edad), clase)


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
        index = self.to_index(value)
        out[index] = 1
        return out

    def to_one_hot_tensor(self, value):
        v = self.to_one_hot_array(value)
        out = torch.from_numpy(v)
        return out

    def list_to_one_hot_tensor(self, l):
        out_list = []
        for val in t:
            v = self.to_one_hot_tensor(val)
            out_list.append(v)
        out = torch.stack(out_list)
        return(out)

    def tensor_to_one_hot_tensor(self, t):
        # TODO: esto puede ser mucho más eficiente!
        input_list = [x.item() for x in t]
        out = self.list_to_one_hot_tensor(input_list)
        return out