import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pipeline import LBP_3DT
import pickle

mapping=pd.read_csv('../Datasets/XV Clinical Data/WCH_XV_genotypes.csv')
mapping.columns=['Record ID','Condition']
mapping=dict(zip(mapping['Record ID'],mapping['Condition']))


csv_dir = '../Datasets/XV Clinical Data'
csv_files = [fold for fold in os.listdir(csv_dir) if os.path.isdir(os.path.join(csv_dir, fold)) and 'WCH' in fold]


dataframes=[]
control=[]
cf=[]
for fold in csv_files:
    p=os.path.join(csv_dir, fold)
    key=int(p.split('-')[2])-10000
    files=[f for f in os.listdir(p) if os.path.isdir(os.path.join(p, f)) and '-LOBAR' not in f and 'WCH' in f]
    curr_csv=[]
    for f in files:
        path=os.path.join(p, f)
        csv=[pd.read_csv(os.path.join(path,c)) for c in os.listdir(path) if c.endswith('_final.csv')]
        if 'EXP' in f:
            csv[0]['Frame']=csv[0]['Frame']+7
        curr_csv.extend(csv)
    

    dataframes.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))
    if 'Control' in mapping[key]:
        control.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))
    else:
        cf.append(pd.merge(curr_csv[0], curr_csv[1], how='outer'))
    
dataframes=[]
for c in control:
    c.columns=['Frame','SV','X','Y','Z']
    dataframes.append([c,0])
for c in cf:
    c.columns=['Frame','SV','X','Y','Z']
    dataframes.append([c,1])

lbp=LBP_3DT(dataframes)

features=lbp.extract()

with open('file', 'wb') as fp:
    pickle.dump(features, fp)
    print('Done writing list into a binary file')




    
    




