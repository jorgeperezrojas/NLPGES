from collections import Counter, OrderedDict
import re
import sys

# TODO: esto debería sacarlo hacia algún archivo general de preocesamiento de texto
precision = 0
suffixes = ['', 'K', 'M', 'G']

def hformat(num):
    _m = sum([abs(num/1000.0**x) >= 1 for x in range(1, len(suffixes))])
    return f'{num/1000.0**_m:.{precision}f}{suffixes[_m]}'

class NameFinder():
    '''
    Clase para encontrar nombres en texto. Usa varias heurísticas, entre ellas, que los nombres
    deben comenzar con mayúsculas. En el constructor se pueden especificar el reemplazo para cada
    nombre y un conjunto de reemplazos especiales como una lista de pares de la forma (nombre, reemplazo), 
    por ejemplo para reemplazar los nombres de enfermedades específicas. También se puede especificar una lista 
    de nombres para mantener (no reemplazar). 

    Cuidado! hace todo el procesamiento en memoria principal.
    '''
    def __read_names_file(self, filename):
        __names = Counter()
        data = open(filename).read()
        lines = data.split('\n')[:-1] # forget the last line
        for line in lines:
            nom = line.split('\t')[0]
            count = int(line.split('\t')[1])
            if count >= self.min_freq:
                __names[nom] = count
        __total_names = sum(__names.values())
        return __names, __total_names

    def __init__(self, 
        min_freq = 200,
        male_name_file = 'models/util_data/nombres_hombres.txt', 
        female_name_file = 'models/util_data/nombres_mujeres.txt', 
        family_name_file = 'models/util_data/apellidos.txt',
        replace_male='<NOMBRE>', replace_female='<NOMBRE>', replace_family='<APELIIDO>',
        special_replace=[],
        ignore_names=[]
        ):

        self.special_replace_names = {(name.lower()):replace for (name,replace) in special_replace}
        self.ignore_names = [name.lower() for name in ignore_names]
        self.min_freq = min_freq
        self.replace_male = replace_male
        self.replace_female = replace_female
        self.replace_family = replace_family

        ### read male names
        self.__male_names, self.__total_male_names = self.__read_names_file(male_name_file)

        ### read female names
        self.__female_names, self.__total_female_names = self.__read_names_file(female_name_file)

        ### all given names
        self.__given_names = self.__male_names + self.__female_names
        self.__total_given_names = sum(self.__given_names.values())

        ### read family names
        self.__family_names, self.__total_family_names = self.__read_names_file(family_name_file)

        ### auxiliary data to parse name strings
        self.__letters = set('aáeéoóíúiuübcdfghjklmnñopqrstvwxyz')
        self.__no_acc_letters = set('abcdefghijklmnñopqrstuvwxyz')
        self.__acc = 'áéíóúàèìòùäëïöü'
        self.__no_acc = 'aeiou'
        self.__no_acc_dict = {}
        for i in range(0, len(self.__acc)):
            self.__no_acc_dict[self.__acc[i]] = self.__no_acc[i%len(self.__no_acc)]


    def replace_names(self, text, verbose=False, to_report=100000, max_name_length=100):

        # first search for contiguous strings of only letters
        i = 0
        output_text = ''
        while i < len(text):
            if verbose and i%to_report == 0:
                partial_info = f'\rReplacing names (processing chars): {hformat(i)}/{hformat(len(text))}       '
                sys.stdout.write(partial_info)
            c = text[i].lower()
            if c not in self.__letters:
                output_text += c
                i += 1
            else:
                # a candidate name has begun
                init = i
                while i < len(text) and text[i].lower() in self.__letters and (i - init) < max_name_length:
                    i += 1
                end = i
                candidate_name = text[init:end]
                if candidate_name[0] in self.__letters: 
                    # the first letter is a lowercase letter, it is assumed that it is not a name
                    replace = text[init:end]
                else:
                    ### replace accent symbols to create a normalized version
                    candidate_name_normalized = candidate_name.lower()
                    candidate_name_normalized = ''.join([c if c in self.__no_acc_letters else self.__no_acc_dict[c] for c in candidate_name_normalized])

                    if candidate_name_normalized in self.ignore_names:
                        replace = text[init:end]
                    elif candidate_name_normalized in self.special_replace_names:
                        replace = self.special_replace_names[candidate_name_normalized]
                    elif candidate_name_normalized in self.__male_names:
                        replace = self.replace_male
                    elif candidate_name_normalized in self.__female_names:
                        replace = self.replace_female
                    elif candidate_name_normalized in self.__family_names:
                        replace = self.replace_family
                    else:
                        replace = text[init:end]

                output_text += replace
        if verbose:
            print()
        return output_text

