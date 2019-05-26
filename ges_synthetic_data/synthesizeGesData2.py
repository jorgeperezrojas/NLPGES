import random
import csv
import re
from collections import defaultdict

EJEMPLOS_PARA_TODAS_LAS_EDADES = True
EJEMPLOS_AL_AZAR_POR_PATOLOGIA = 60

random.seed(333)

# Funcion ultra simple para normalizar (un poco el texto)
# se puede mejorar con un tokenizador o algo así.
def normalizaTextoPatologia(text):
    keep = set('aáeéoóíúiuübcdfghjklmnñopqrstvwxyz ')
    text = text.lower()
    text = ''.join([c for c in text if c in keep])
    text = re.sub(' +',' ',text)
    return text

# Lee los datos de la lista de patologías y asocia los datos mas 
# una lista de intervalos a cada patología.
# Asume que los rangos están especificados con min y max y que corresponden
# al intervalo de números enteros [min,max), es decir, min se incluye en el intervalo
# pero max se excluye. Los números incluídos son los mismos en range(min, max)
patologias_data = defaultdict(dict)
patologias_rangos = defaultdict(list)
with open('ges-health-problems.csv') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        texto_original = row['included_pathology']
        key = normalizaTextoPatologia(texto_original)
        patologias_data[key]['especialidad'] = 'None' # en el futuro row['especialidad']
        patologias_data[key]['texto_original'] = texto_original
        left = int(row['min'])
        right = int(row['max'])
        patologias_rangos[key].append(range(left,right))

# Escribe los casos sintéticos.
with open('super-ges.csv', 'w') as outfile:
    writer = csv.DictWriter(outfile, ['SOSPECHA_DIAGNOSTICA', 'ESPECIALIDAD', 'EDAD', 'GES'])
    writer.writeheader()
    for key in patologias_data:
        row = {}
        rangos = patologias_rangos[key]
        row['SOSPECHA_DIAGNOSTICA'] = patologias_data[key]['texto_original']
        row['ESPECIALIDAD'] = patologias_data[key]['especialidad']
        possible_ages = range(0,100)
        
        ges_ages = set.union(*[set(rango) for rango in rangos])
        ges_ages = list(ges_ages)
        not_ges_ages = [x for x in possible_ages if x not in ges_ages]

        # Crea un ejemplo por cada edad:
        if EJEMPLOS_PARA_TODAS_LAS_EDADES:
            for i in range(0,100):
                row['EDAD'] = i
                if i in ges_ages:
                    row['GES'] = 'True'
                else:
                    row['GES'] = 'False'
                writer.writerow(row)

        # Crea ejemplos adicionales al azar
        for i in range(EJEMPLOS_AL_AZAR_POR_PATOLOGIA):
            # Si not_ges_ages es vacío entonces la patología es siempre GES.
            # De otra forma, decide con 50% prob cuando hacer un ejemplo positivo para clase GES
            if (not_ges_ages == []) or (random.uniform(0,1) > 0.5):
                row['GES'] = 'True'
                row['EDAD'] = random.choice(ges_ages)
            else:
                row['GES'] = 'False'
                row['EDAD'] = random.choice(not_ges_ages)
            writer.writerow(row)