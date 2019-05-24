import numpy as np
import pandas as pd
import random
import os
def getRange(row):
    return list(np.arange(row['min'], row['max']+1))
data = pd.read_excel('data/ges-health-problems.xlsx')
data['range'] = data.apply (lambda row: getRange(row), axis=1)
expanded = []
for index, row in data.iterrows():
    for i in range(100):
        ges = bool((np.where(np.random.random() > 0.5,True,False)))
        ges_ages = list(set(row.range))
        not_ges_ages = [x for x in list(range(0,101)) if x not in ges_ages]
        if ges:
            age = random.choice(ges_ages)
        elif len(not_ges_ages)>0:
            age = random.choice(not_ges_ages)
        else:
            age = random.choice(list(range(0,101)))
            ges = True
        expanded.append([row['included_pathology'],age,ges])
expanded_df = pd.DataFrame(expanded, columns=['SOSPECHA_DIAGNOSTICA','edad', 'GES'])
expanded_df.to_excel('data/super-ges.xlsx')