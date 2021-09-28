from __future__ import print_function
import pickle
from model.toolkits.interactions import close_contacts
from model.toolkits.interactions import hydrophobic_contacts
from model.toolkits.interactions import salt_bridges
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
from model.IFP import cal_Interactions
from pathos.multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="dataset to train",
                        default='config.txt')
    parser.add_argument("--save", help="name of dataframe of interactions",
                        default='df_interaction')
    parser.add_argument("--n_jobs", help="number of threads", type=int,
                        default=10)
    args = parser.parse_args()
    return args


def save_obj(obj, name):
    os.system('mkdir obj')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def walk_folder(path, suffix):
    processed = []
    files = os.listdir(path)
    print(f"{len(files)} files have been detected!")
    for file in files:
        # print(file)
        if suffix in file:
            base_name = os.path.basename(file)
            # print(base_name)
            simple_name = base_name.replace('_', '.').split('.')
            simple_name = simple_name[0]
            processed.append({'simple_name': simple_name,
                              'base_name': base_name, 'full_name': file})
    # print(processed)
    return processed


def concat_df(dic_main, dic):
    # df_interaction = ''
    # for i in range(len(df_res)):
    #     if i == 0:
    #         df_interaction = df_res[0]
    #     else:
    #         for key in df_interaction.keys():
    #             df_interaction[key] = pd.concat(
    #                 [df_interaction[key], df_res[i][key]])
    # return df_interaction
    for key in dic_main.keys():
        if len(dic_main[key]) == 0:
            dic_main[key] = dic[key]
        else:
            dic_main[key] = pd.concat([dic_main[key], dic[key]])
    return dic_main


def cal_interacions(ligand, mol_p, config):
    ligand_name = ligand['simple_name']
    ligand = parse_ligand_vina(os.path.join(
        ligand_folder, ligand['base_name']))

    mol_l = Molecule(ligand['docked_ligands'][0], protein=False)
    df_res = cal_Interactions(
        mol_p, mol_l, config)
    print(f"\n{ligand_name}\n")
    for key in df_res.keys():
        df_res[key]['Molecule'] = ligand_name
    return df_res
    #     df_Interaction = concat_df(df_Interaction, df_res)


def main(config, args):
    debug = 0  # if in debug model 0: False  1: True
    df_Interaction = {'df_hbond': '', 'df_halogen': '',
                      'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
    protein = config['protein']
    ligand_folder = config['ligand_folder']
    ligands = walk_folder(ligand_folder, '_out.pdbqt')
    print(protein)
    protein = parse_protein_vina(protein)
    mol_p = Molecule(protein['protein'], protein=True)
    # with Pool(args.n_jobs) as pool:
    #     cal_interacions_p = partial(
    #         cal_interacions, mol_p=mol_p, config=config)
    #     df_res = [x for x in tqdm(
    #         pool.map(cal_interacions_p, list(ligands)),
    #         total=len(ligands),
    #         miniters=50
    #     )
    #         if x is not None]
    # df_Interaction = concat_df(df_res)
    for ligand in ligands:
        ligand_name = ligand['simple_name']
        ligand = parse_ligand_vina(os.path.join(
            ligand_folder, ligand['base_name']))
        mol_l = Molecule(ligand['docked_ligands'][0], protein=False)
        df_res = cal_Interactions(
            mol_p, mol_l, config)
        print(f"\n\n{ligand_name}\n\n")
        for key in df_res.keys():
            df_res[key]['Molecule'] = ligand_name
        df_Interaction = concat_df(df_Interaction, df_res)
    save_obj(df_Interaction, args.save)


if __name__ == "__main__":
    args = get_parser()
    config = parse_config_vina(args.config)
    main(config, args)
