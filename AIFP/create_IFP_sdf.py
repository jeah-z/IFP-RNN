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
try:
    import pybel
except ImportError:
    from openbabel import pybel
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
from model.IFP import cal_Interactions, get_Molecules, cal_IFP
from pathos.multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from pathlib import Path
from joblib import Parallel, delayed
import threading
import math


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
    ligand = parse_ligand_vina(os.path.join(ligand_folder,
                                            ligand['base_name']))
    ligand_poses = ligand['docked_ligands']
    interaction_poses = []
    for ipose in range(len(ligand_poses)):
        mol_l = Molecule(ligand_poses[ipose], protein=False)
        df_res = cal_Interactions(mol_p, mol_l, config)
        print(f"\n{ligand_name}-{ipose}\n")
        for key in df_res.keys():
            df_res[key]['Molecule'] = f'{ligand_name}-{ipose}'
        interaction_poses.append(df_res)
    return interaction_poses
    #     df_Interaction = concat_df(df_Interaction, df_res)


def IFP(ligand, config):
    '''The  interaction types of different ligand poses will be transformed into interaction fingerprint.
    '''
    protein = read_protein()
    mol_p = Molecule(protein, protein=True)
    base_name = os.path.basename(ligand)

    # print(base_name)
    simple_name = base_name.replace('_out.pdbqt', '')
    processed = {
        'simple_name': simple_name,
        'base_name': base_name,
        'full_name': ligand
    }
    interaction_poses = cal_interacions_run(processed, mol_p, config)
    reference_atom = load_obj('refer_atoms_list')
    reference_res = load_obj('refer_res_list')
    IFP_poses = []
    AAIFP, RESIFP = cal_IFP(interaction_poses[0], reference_atom,
                            reference_res)
    AAIFP = [f'{simple_name}'] + AAIFP
    RESIFP = [f'{simple_name}'] + RESIFP
    IFP_poses = [[AAIFP, RESIFP]]
    return IFP_poses


def read_protein():
    '''Read protein from the ./obj/protein.pdbqt'''
    convert = ob.OBConversion()
    convert.SetInFormat('pdbqt')
    protein = ob.OBMol()
    convert.ReadFile(protein, './obj/protein.pdbqt')
    return protein


def mol2_pdbqt(mols, config, thread=0):
    if thread == 0:
        process_bar = ShowProcess(len(mols), 'Processing sdf was done!')
    results = []
    for i, ligand in enumerate(mols):
        try:
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
            result = [
                f'{ligand.title}_{ligand.data["Pose_id"]}',
                ligand.data["Docking_score"], ligand.data["SMILES"],
                ligand.data["Pose_id"]
            ]
            # print(result)
            results.append(result)
        except Exception as e:
            print(e)
            print(f"Above abnormity encountered!")
            continue
    return results


class MyThread(threading.Thread):
    ''' Obtain the results of the different threads!
    '''
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func  # Function for parallel running
        self.args = args  # Parameters

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        return self.result


def main(config, args):
    debug = 0  # if in debug model 0: False  1: True
    df_Interaction = {
        'df_hbond': '',
        'df_halogen': '',
        'df_elecpair': '',
        'df_hydrophobic': '',
        'df_pistack': ''
    }
    # protein = read_protein()
    # mol_p = Molecule(protein, protein=True)
    print("Processing the sdf!")
    sdf_path = Path(args.sdf).absolute()
    ligands = list(pybel.readfile('sdf', str(sdf_path)))
    # ligands = ligands[:5000] # Speed up the test only!
    temp_folder = f'Tmp_{Path(args.sdf).stem}'
    os.system(f'mkdir {temp_folder}')
    os.chdir(f"{temp_folder}")
    # res_list = Parallel(n_jobs=50)(delayed(mol2_pdbqt)(ligand, config)
    #                                for ligand in tqdm(ligands))
    threads = []
    job_each_thread = math.floor(len(ligands) / args.n_jobs)
    for i in range(args.n_jobs):
        t = MyThread(mol2_pdbqt,
                     (ligands[i * job_each_thread:(i + 1) * job_each_thread],
                      config, i))
        threads.append(t)
    for t in threads:
        t.start()
    results = []
    for t in threads:
        t.join()
        try:
            if t.get_result() != None:
                results += t.get_result()
        except Exception as e:
            print(e)
            continue

    df_info = pd.DataFrame(
        results, columns=['Name', 'Docking_score', 'SMILES', 'Pose_id'])
    df_info.to_csv(f'info.csv', index=None)
    os.chdir(f"../")
    ## Processing sdf of docking results finished!

    config['ligand_folder'] = ligand_folder = temp_folder
    ligands = walk_folder(ligand_folder, '_out.pdbqt')
    with Pool(args.n_jobs) as pool:
        IFP_p = partial(IFP, config=config)
        res_list = [
            x for x in tqdm(pool.imap(IFP_p, list(ligands)),
                            total=len(ligands),
                            miniters=50) if x is not None
        ]
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

    AAIFP_full = pd.DataFrame(AAIFP_full, columns=colname)
    colname = ['Molecule']
    for ires in reference_res:
        for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
            colname.append(f'{ires}-{iifp}')
    ResIFP_full = pd.DataFrame(ResIFP_full, columns=colname)
    AAIFP_full.to_csv(f'{args.save}_AAIFP.csv', index=None)
    ResIFP_full.to_csv(f'{args.save}_ResIFP.csv', index=None)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="config", default='config.txt')
    parser.add_argument("--save", help="dataframe name of IFP", default='IFP')
    parser.add_argument("--n_jobs",
                        help="number of threads",
                        type=int,
                        default=10)
    parser.add_argument("--sdf",
                        help="sdf file of docking results.",
                        default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    config = parse_config_vina(args.config)
    main(config, args)