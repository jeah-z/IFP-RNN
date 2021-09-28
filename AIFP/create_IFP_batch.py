from __future__ import print_function
import pickle
from model.toolkits.interactions import close_contacts
from model.toolkits.interactions import hydrophobic_contacts
from model.toolkits.interactions import salt_bridges
import numpy as np
import pandas as pd
# from bitarray import bitarray
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
import sys
import os
import argparse
# from time import time
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
from model.IFP import cal_Interactions,  get_Molecules, cal_IFP
from pathos.multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config",
                        default='config.txt')
    parser.add_argument("--save", help="dataframe name of IFP",
                        default='IFP')
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
    outpdbqt = []
    for file in files:
        # print(file)
        if suffix in file:
            outpdbqt.append(file)
    #         base_name = os.path.basename(file)
    #         # print(base_name)
    #         simple_name = base_name.replace('_', '.').split('.')
    #         simple_name = simple_name[0]
    #         processed.append({'simple_name': simple_name,
    #                           'base_name': base_name, 'full_name': file})
    # # print(processed)
    return outpdbqt


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


def cal_interacions_run(ligand, mol_p, config):
    ''' This funciton will calculate all the intercations between the protein and all the docking poses of the ligand.
    '''
    ligand_name = ligand['simple_name']
    ligand_folder = config['ligand_folder']
    ligand = parse_ligand_vina(os.path.join(
        ligand_folder, ligand['base_name']))
    ligand_poses = ligand['docked_ligands']
    interaction_poses = []
    for ipose in range(len(ligand_poses)):
        mol_l = Molecule(ligand_poses[ipose], protein=False)
        df_res = cal_Interactions(
            mol_p, mol_l, config)
        print(f"\n{ligand_name}-{ipose}\n")
        for key in df_res.keys():
            df_res[key]['Molecule'] = f'{ligand_name}-{ipose}'
        interaction_poses.append(df_res)
    return interaction_poses
    #     df_Interaction = concat_df(df_Interaction, df_res)


def IFP(ligand, config):
    '''The  interaction types of different ligand poses will be transformed into interaction fingerprint.
    '''
    protein = config['protein']
    protein = parse_protein_vina(protein)
    mol_p = Molecule(protein['protein'], protein=True)
    base_name = os.path.basename(ligand)

    # print(base_name)
    simple_name = base_name.replace('_out.pdbqt', '')
    processed = {'simple_name': simple_name,
                 'base_name': base_name, 'full_name': ligand}
    interaction_poses = cal_interacions_run(processed, mol_p, config)
    reference_atom = load_obj('refer_atoms_list')
    reference_res = load_obj('refer_res_list')
    IFP_poses = []
    for ipose in range(len(interaction_poses)):
        if ipose < 5:
            # the IFPs are python list.
            AAIFP, RESIFP = cal_IFP(
                interaction_poses[ipose], reference_atom, reference_res)
            AAIFP = [f'{simple_name}_{ipose}']+AAIFP
            RESIFP = [f'{simple_name}_{ipose}']+RESIFP
            IFP_poses.append([AAIFP, RESIFP])
    return IFP_poses


def main(config, args):
    debug = 0  # if in debug model 0: False  1: True
    df_Interaction = {'df_hbond': '', 'df_halogen': '',
                      'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
    protein = config['protein']
    ligand_folder = config['ligand_folder']
    ligands = walk_folder(ligand_folder, '_out.pdbqt')
    protein = parse_protein_vina(protein)
    mol_p = Molecule(protein['protein'], protein=True)

    with Pool(args.n_jobs) as pool:
        IFP_p = partial(
            IFP,  config=config)
        res_list = [x for x in tqdm(
            pool.imap(IFP_p, list(ligands)),
            total=len(ligands),
            miniters=50
        )
            if x is not None]
    AAIFP_full = []
    ResIFP_full = []
    for IFP_poses in res_list:
        for iPose in IFP_poses:
            AAIFP_full.append(iPose[0])
            ResIFP_full.append(iPose[1])

    # df_Interaction = concat_df(df_res)
    reference_atom = load_obj('refer_atoms_list')
    reference_res = load_obj('refer_res_list')
    colname = ['Molecule']
    for iatm in reference_atom:
        for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
            colname.append(f'{iatm}-{iifp}')

    AAIFP_full = pd.DataFrame(
        AAIFP_full, columns=colname)
    colname = ['Molecule']
    for ires in reference_res:
        for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
            colname.append(f'{ires}-{iifp}')
    ResIFP_full = pd.DataFrame(
        ResIFP_full, columns=colname)
    AAIFP_full.to_csv(f'{args.save}_AAIFP.csv', index=None)
    ResIFP_full.to_csv(f'{args.save}_ResIFP.csv', index=None)


if __name__ == "__main__":
    args = get_parser()
    config = parse_config_vina(args.config)
    main(config, args)
