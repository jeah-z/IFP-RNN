# %%
from __future__ import print_function
import pickle
import numpy as np
import pandas as pd
from bitarray import bitarray
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
import sys
import os
import argparse
from time import time
# from timeout import timeout
from model.toolkits.parse_conf import parse_config_vina, parse_protein_vina, parse_ligand_vina
# from model.toolkits.parse_conf import parse_config
# from model.toolkits.parse_docking_conf import parse_vina_conf
from model.toolkits.PARAMETERS import HYDROPHOBIC, AROMATIC, HBOND, ELECTROSTATIC, HBOND_ANGLE,\
    AROMATIC_ANGLE_LOW, AROMATIC_ANGLE_HIGH
from model.obbl import Molecule
from model.toolkits.spatial import angle, distance
from model.toolkits.interactions import hbonds, pi_stacking, salt_bridges, \
    hydrophobic_contacts, close_contacts, halogenbonds
from model.toolkits.pocket import pocket_atoms
# from model.IFP import cal_Interactions
from pebble import concurrent, ProcessPool
from concurrent.futures import TimeoutError

# %%


def cal_Interactions(mol_p, mol_l, config):
    '''
    The different type interacitons  between specific pose of ligand and protein will be calculated.  The dictionary of interactions will be outputed as as dictionary.
    '''
    # mol_l = Molecule(OBMol=ligand)
    # mol_p = Molecule(OBMol=protein)

    # Hbonds detection
    print('\n'+'#'*10+'\tHbonds detection\t'+'#'*10)
    ad_list, da_list, strict_list = hbonds(
        mol_p, mol_l, cutoff=float(config['hbond_cutoff']))
    hbd_dict = {'ad_list': ad_list,
                'da_list': da_list, 'strict_list': strict_list}
    hbd_df = pd.DataFrame.from_dict(hbd_dict)
    print(hbd_df)
    Hbonds = []  # create_empty_list(12)
    for i in range(len(ad_list)):
        d = distance([ad_list[i]['coords']], [da_list[i]['coords']])
        new_line = [ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
                    ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13], '', ad_list[i][11], ad_list[i][10]]
        Hbonds.append(new_line)  # = list_append2d(Hbonds, new_line)

        print("Id: %s %s\t AtomicNum: %s %s\t Distance: %s \t IsDonor: %s %s\t IsAcceptor: %s %s\t" %
              (ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
               ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13]))
    df_hbond = pd.DataFrame(
        Hbonds, columns=['Id_p', 'Id_l', 'AtomicNum_p', 'AtomicNum_l', 'Distance',
                         'IsDonor_p', 'IsDonor_l', 'IsAcceptor_p', 'IsAcceptor_l', 'Molecule', 'ResName_p', 'ResNum_p'])

    # Halogenbonds detection
protein = '/mnt/home/zhangjie/Projects/cRNN/CDK2/Data/cdk2.pdbqt'
ligand = '/mnt/home/zhangjie/Projects/cRNN/CDK2/Data/cdk2Crystal_dock/1KE8_out.pdbqt'
config_file = '/mnt/home/zhangjie/Projects/cRNN/CDK2/Data/config_ifp.txt'
config = parse_config_vina(config_file)
protein = parse_protein_vina(protein)
mol_p = Molecule(protein['protein'], protein=True)
ligand = parse_ligand_vina(ligand)
ligand_pose1 = ligand['docked_ligands'][0]
mol_l = Molecule(ligand_pose1, protein=False)
# df_res = cal_Interactions(
#     mol_p, mol_l, config)


# Hbonds detection
print('\n'+'#'*10+'\tHbonds detection\t'+'#'*10)
ad_list, da_list, angle_list = hbonds(
    mol_p, mol_l, cutoff=float(config['hbond_cutoff']))
Hbonds = []  # create_empty_list(12)
for i in range(len(ad_list)):
    d = distance([ad_list[i]['coords']], [da_list[i]['coords']])
    new_line = [ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
                ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13], '', ad_list[i][11], ad_list[i][10]]
    Hbonds.append(new_line)  # = list_append2d(Hbonds, new_line)

    print("Id: %s %s\t AtomicNum: %s %s\t Distance: %s \t IsDonor: %s %s\t IsAcceptor: %s %s\t" %
          (ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
           ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13]))
df_hbond = pd.DataFrame(
    Hbonds, columns=['Id_p', 'Id_l', 'AtomicNum_p', 'AtomicNum_l', 'Distance',
                     'IsDonor_p', 'IsDonor_l', 'IsAcceptor_p', 'IsAcceptor_l', 'Molecule', 'ResName_p', 'ResNum_p'])

# %%
# print(strict_list)
print('ad_list'+'#'*20)
# print(ad_list['coords'])
print(ad_list)

print('da_list'+'#'*20)
print(da_list)
print(f'angle_list={angle_list}')
# %%
