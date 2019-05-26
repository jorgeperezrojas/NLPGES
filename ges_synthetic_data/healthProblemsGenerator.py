# este programa genera una planilla en donde cada fila es
# una patología. Una columna es la patología misma y
# otra columna es el problema de salud a la cual pertenece
import json
import pandas as pd
def js_r(filename):
    with open(filename, encoding='utf-8') as f_in:
        return(json.load(f_in))
all_pats = js_r('ges-health-problems-1557977580.txt')
all_pats_list = []
for key,val in all_pats.items():
    for pat in val:
        all_pats_list.append(tuple([key]+[pat]))
pd.DataFrame(all_pats_list, columns=['healt_problem','included_pathology']).to_excel('ges-health-problems.xlsx')
problems = all_pats.keys()