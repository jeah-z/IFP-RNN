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
try:
    import pybel
except ImportError:
    from openbabel import pybel
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
from pathlib import Path
from joblib import Parallel, delayed
import threading
import math


def save_obj(obj, name):
    os.system('mkdir obj')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


class ShowProcess():
    """
    显示处理进度的类
    调用该类相关函数即可实现处理进度的显示
    """
    i = 0  # 当前的处理进度
    max_steps = 0  # 总共需要处理的次数
    max_arrow = 50  #进度条的长度
    infoDone = 'done'

    # 初始化函数，需要知道总共的处理次数
    def __init__(self, max_steps, infoDone='Done'):
        self.max_steps = max_steps
        self.i = 0
        self.infoDone = infoDone

    # 显示函数，根据当前的处理进度i显示进度
    # 效果为[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]100.00%
    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)  #计算显示多少个'>'
        num_line = self.max_arrow - num_arrow  #计算显示多少个'-'
        percent = self.i * 100.0 / self.max_steps  #计算完成进度，格式为xx.xx%
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r' #带输出的字符串，'\r'表示不换行回到最左边
        sys.stdout.write(process_bar)  #这两句打印字符到终端
        sys.stdout.flush()
        if self.i >= self.max_steps:
            self.close()

    def close(self):
        print('')
        print(self.infoDone)
        self.i = 0


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
            processed.append({
                'simple_name': simple_name,
                'base_name': base_name,
                'full_name': file
            })
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
    # print(f'{ligand_name} is being processing!')
    ligand = parse_ligand_vina(
        os.path.join(refer_ligands_folder, ligand['base_name']))
    ligand_poses = ligand['docked_ligands']
    PocketAtoms_mult = []
    for ipose in range(len(ligand_poses)):
        mol_l = Molecule(ligand_poses[ipose], protein=False)
        l_atomdict = mol_l.atom_dict
        PocketAtoms = pocket_atoms(p_atomdict, l_atomdict, 6.0)
        # print(f'pocket_atoms: \n {PocketAtoms}')
        PocketAtoms_mult = PocketAtoms_mult + list(PocketAtoms)
        PocketAtoms_mult = list(set(PocketAtoms_mult))
    return PocketAtoms_mult


def read_protein(protein_file):
    protein_path = Path(protein_file)
    suffix = protein_path.suffix
    print(
        f"Reading protin: \n Protein file: {protein_path} \n File type: {suffix}"
    )
    convert = ob.OBConversion()
    convert.SetInFormat('pdbqt')
    protein = ob.OBMol()
    convert.ReadFile(protein, './obj/protein.pdbqt')
    return protein


def mol2_pdbqt(mols, config, thread=0):
    if thread == 0:
        process_bar = ShowProcess(len(mols), 'Processing sdf was done!')
    for i, ligand in enumerate(mols):
        # print(ligand.title)
        ligand.write('mol2',
                     f'{ligand.title}_{ligand.data["Pose_id"]}.mol2',
                     overwrite=True)
        os.system(
            f'{Path(config["prepare_ligand4"]).parent/"prepare_ligand4.py"} -l {ligand.title}_{ligand.data["Pose_id"]}.mol2 -o {ligand.title}_{ligand.data["Pose_id"]}_out.pdbqt'
        )
        os.system(f'rm {ligand.title}_{ligand.data["Pose_id"]}.mol2')
        if thread == 0:
            process_bar.show_process()


def main(args):
    config = parse_config_vina(args.config)
    os.system(
        f'{Path(config["prepare_ligand4"]).parent/"prepare_receptor4.py"} -r {args.protein} -o ./obj/protein.pdbqt'
    )
    protein = read_protein(args.protein)
    print("Processing the sdf!")
    sdf_path = Path(args.sdf).absolute()
    ligands = list(pybel.readfile('sdf', str(sdf_path)))
    # ligands = ligands[:5000] # Speed up the test only!
    temp_folder = f'.Tmp_{Path(args.sdf).stem}'
    os.system(f'mkdir {temp_folder}')
    os.chdir(f"{temp_folder}")
    # res_list = Parallel(n_jobs=50)(delayed(mol2_pdbqt)(ligand, config)
    #                                for ligand in tqdm(ligands))
    threads = []
    job_each_thread = math.floor(len(ligands) / args.n_jobs)
    for i in range(args.n_jobs):
        t = threading.Thread(target=mol2_pdbqt,
                             args=(ligands[i * job_each_thread:(i + 1) *
                                           job_each_thread], config, i))
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    os.chdir(f"../")

    debug = 0  # if in debug model 0: False  1: True
    df_Interaction = {
        'df_hbond': '',
        'df_halogen': '',
        'df_elecpair': '',
        'df_hydrophobic': '',
        'df_pistack': ''
    }
    refer_ligands_folder = temp_folder
    ligands = walk_folder(refer_ligands_folder, '_out.pdbqt')
    mol_p = Molecule(protein, protein=True)
    # msms_path = '/mnt/home/zhangjie/Projects/cRNN/AIFP/model/toolkits/msms'
    # SurfaceAtoms = surface_atoms(config['protein'], msms_path)
    # print(f'\nNumber of surface atoms:  {len(SurfaceAtoms)}\n')
    # print(f'\nSurface atoms: \n {SurfaceAtoms}\n')
    PocketAtoms_mult = []
    p_atomdict = mol_p.atom_dict
    print(f'\nNumber of ligands used to create reference: {len(ligands)}\n')
    with Pool(args.n_jobs) as pool:
        detect_pocketAtom_p = partial(
            detect_pocketAtom,
            p_atomdict=p_atomdict,
            refer_ligands_folder=refer_ligands_folder)
        pAtom_list = [
            x for x in tqdm(pool.imap(detect_pocketAtom_p, list(ligands)),
                            total=len(ligands),
                            miniters=50) if x is not None
        ]
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
    os.system(f'rm -rf {temp_folder}')  # Clean the temp results


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                        help="ifp config file",
                        default='config.txt')
    parser.add_argument("--n_jobs",
                        help="number of threads",
                        type=int,
                        default=40)
    parser.add_argument(
        "--protein",
        help=
        "Protein file used for docking! The file type will be detected based on the suffix.",
        default='')
    parser.add_argument("--sdf",
                        help="sdf file of docking results.",
                        default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
