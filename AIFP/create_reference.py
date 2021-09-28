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
from model.toolkits.pocket import pocket_atoms, surface_atoms
from model.IFP import cal_Interactions
from pathos.multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="ifp config file",
                        default='config.txt')
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
    for key in dic_main.keys():
        if len(dic_main[key]) == 0:
            dic_main[key] = dic[key]
        else:
            dic_main[key] = pd.concat([dic_main[key], dic[key]])
    return dic_main


def detect_pocketAtom(ligand, p_atomdict, refer_ligands_folder):
    ligand_name = ligand['simple_name']
    print(f'{ligand_name} is being processing!')
    ligand = parse_ligand_vina(os.path.join(
        refer_ligands_folder, ligand['base_name']))
    ligand_poses = ligand['docked_ligands']
    PocketAtoms_mult = []
    for ipose in range(len(ligand_poses)):
        mol_l = Molecule(ligand_poses[ipose], protein=False)
        l_atomdict = mol_l.atom_dict
        PocketAtoms = pocket_atoms(p_atomdict, l_atomdict, 6.0)
        # print(f'pocket_atoms: \n {PocketAtoms}')
        PocketAtoms_mult = PocketAtoms_mult+list(PocketAtoms)
        PocketAtoms_mult = list(set(PocketAtoms_mult))
    return PocketAtoms_mult


def main(config):
    debug = 0  # if in debug model 0: False  1: True
    df_Interaction = {'df_hbond': '', 'df_halogen': '',
                      'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
    protein = config['protein']
    refer_ligands_folder = config['refer_ligands_folder']
    ligands = walk_folder(refer_ligands_folder, '_out.pdbqt')
    protein = parse_protein_vina(protein)
    mol_p = Molecule(protein['protein'], protein=True)
    # msms_path = '/mnt/home/zhangjie/Projects/cRNN/AIFP/model/toolkits/msms'
    # SurfaceAtoms = surface_atoms(config['protein'], msms_path)
    # print(f'\nNumber of surface atoms:  {len(SurfaceAtoms)}\n')
    # print(f'\nSurface atoms: \n {SurfaceAtoms}\n')
    PocketAtoms_mult = []
    p_atomdict = mol_p.atom_dict
    print(
        f'\nNumber of ligands used to create reference: {len(ligands)}\n')
    with Pool(40) as pool:
        detect_pocketAtom_p = partial(
            detect_pocketAtom, p_atomdict=p_atomdict, refer_ligands_folder=refer_ligands_folder)
        pAtom_list = [x for x in tqdm(
            pool.imap(detect_pocketAtom_p, list(ligands)),
            total=len(ligands),
            miniters=50
        )
            if x is not None]
    for iPatom in pAtom_list:
        PocketAtoms_mult.extend(iPatom)
    PocketAtoms_mult = list(set(PocketAtoms_mult))
    print(f'\nPocketAtoms_mult: \n {PocketAtoms_mult}\n')

    # refer_atoms = [x for x in PocketAtoms_mult if x in SurfaceAtoms]
    refer_atoms = PocketAtoms_mult
    refer_atoms.sort()
    print(f'\nNumber of reference atoms:  {len(refer_atoms)}\n')
    print(f'\nrefer_atoms:\n  {refer_atoms}\n')
    refer_res = []
    for atmidx in refer_atoms:
        res_name = p_atomdict[atmidx][11]
        res_num = p_atomdict[atmidx][10]
        refer_res.append(f'{res_name}_{res_num}')
    refer_res = list(set(refer_res))
    refer_res.sort()

    save_obj(refer_atoms, 'refer_atoms_list')
    save_obj(refer_res, 'refer_res_list')
    print(f'\nNumber of reference residues:  {len(refer_res)}\n')
    print(f'\nrefer_res:\n {refer_res}\n')


if __name__ == "__main__":
    args = get_parser()
    config = parse_config_vina(args.config)
    main(config)
