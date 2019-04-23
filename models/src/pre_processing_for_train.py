import re
from collections import Counter
from emoji import UNICODE_EMOJI
import unidecode
from models.src.name_utils import NameFinder
import sys

# TODO: esto debería sacarlo hacia algún archivo general de preocesamiento de texto
delimiters = '<>'
punctuation = '.,\n/?'
map_punctuation = {'¿': '<ai>', '?': '<ci>', '.': '<pt>', '\n': '<nl>', ',': '<cm>', '/': '<br>'}
letras = set('$aáeéoóíúiuübcdfghjklmnñopqrstvwxyz')
max_repetitions = {
    'a':1,'e':1,'i':1,'o':1,'u':1, 
    'c':2, 'd':1, 'f':1, 'g':1, 'h':1, 'j':1, 'k':1, 'l': 2, 'm':1, 
    'n':1, 'p':1, 'q':1, 'r':2, 's':1, 't':1, 'v':1, 'w':1, 'x':1, 'y':1, 'z':1}

acc_chars = set(punctuation).union(letras).union(delimiters)

precision = 0
suffixes = ['', 'K', 'M', 'G']

def hformat(num):
    _m = sum([abs(num/1000.0**x) >= 1 for x in range(1, len(suffixes))])
    return f'{num/1000.0**_m:.{precision}f}{suffixes[_m]}'
# TODO: hasta acá

class Tokenizer():
    '''
    Be careful! remove_contiguous_chars==True can be extremely slow 
    TODO: optimize the deletion of contiguous chars.
    '''
    def __init__(self, voc=None, 
            remove_names=False, 
            special_replace=[], 
            remove_accents=True, 
            remove_contiguous_chars=True,
            **name_finder_kwargs):
        self.remove_names = remove_names
        self.remove_accents = remove_accents
        self.remove_contiguous_chars = remove_contiguous_chars  

        if remove_contiguous_chars:
            # create regular expressions
            self.regex_char_list = []
            for char in max_repetitions:
                regex = char * max_repetitions[char] + '+'
                self.regex_char_list.append((regex,char))

        if remove_names:
            self.nfinder = NameFinder(special_replace=special_replace, **name_finder_kwargs)

    def tokenize(self, text, verbose=False, to_report=1000000):
        if self.remove_names:
            text = self.nfinder.replace_names(text, verbose)

        text = text.lower()

        # elimina puntuación y pon espacios donde se necesite
        char_tokens = []
        for i,c in enumerate(text):
            if verbose and i%to_report == 0:
                partial_info = f'\rProcessing chars to find tokens: {hformat(i)}/{hformat(len(text))}       '
                sys.stdout.write(partial_info)
            to_append = ''
            if c in letras or c == ' ' or c in delimiters:
                if self.remove_accents:
                    c = unidecode.unidecode(c)
                to_append = c
            elif c in UNICODE_EMOJI:
                to_append = ' ' + c + ' '
            elif c in punctuation:
                to_append = ' ' + map_punctuation[c] + ' '

            char_tokens.append(to_append)

        if verbose:
            print('\nGenerating tokens...')  

        text = re.sub(' +',' ',''.join(char_tokens)).strip()
        tokens = text.split(' ')

        # free some memory
        del char_tokens
        del text

        if self.remove_contiguous_chars:
            new_tokens = []
            for i,token in enumerate(tokens):
                if verbose:
                    partial_info = f'\rPost processing tokens (removing contiguous chars): {hformat(i)}/{hformat(len(tokens))}       '
                    sys.stdout.write(partial_info)
                for (regex,char) in self.regex_char_list:
                    token = re.sub(regex,char,token)
                new_tokens.append(token)
            tokens = new_tokens
            if verbose:
                print()

        return tokens

    def generate_voc(self, corpus, min_freq=1, verbose=False):
        word_tokens = self.tokenize(corpus, verbose)
        word_cnt = Counter(word_tokens)
        self.voc = word_cnt
        if min_freq > 1:
            word_cnt = Counter({w:word_cnt[w] for w in word_cnt if word_cnt[w] >= min_freq})
        return word_cnt

    def tokenize_with_voc(self, text, voc=None, unk='<unk>'):
        if voc == None:
            voc = self.voc
        tokens = self.tokenize(text)
        output = [x if x in voc else unk for x in tokens]
        return output

