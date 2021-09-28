from __future__ import print_function
import pickle
import pandas as pd


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


pickleFile = './Results/atomdScorePP/atomdScorePP_234actfullBits_0.4_ifp'
df = load_obj(pickleFile)
# print(f'seed SMILES: {df[0]["seedSmi"]}')
print(f'generated SMILES: {len(df[0]["smis"])}')
print(f'dfRES: {df[1]["dfRes"].columns}')
print(type(df[0]))
seed1 = df[0]
seed2 = df[1]
print('seed1_smi', seed1['seedSmi'])
print('seed2_smi', seed2['seedSmi'])
df_seed1 = seed1['dfRes']
df_seed2 = seed2['dfRes']
df_seed2.to_csv('dfseed2.csv')
