import sys
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import argparse
from pathlib import Path
from joblib import Parallel, delayed
import time


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


def read_smi(smi_file):
    # '''
    # Read SMILES files with names, for example:
    # O=C([O-])CC1S/C(=N\N=C\c2ccc(N3CCOCC3)cc2)NC1=O ZINC000004138686
    # '''
    with open(smi_file, 'r') as f:
        lines = [
            line.split() for line in f.readlines() if len(line.split()) == 2
        ]
    df = pd.DataFrame(lines, columns=['SMILES', 'Name'])
    return df


def prepare_sdf_glide(args):
    '''
    Prepare the docking results of glide in SDF format for creating the IFP.
    1. The name of the compounds in the SDF file and smi file should be the same.
    2. If the SMILES file is not available, the SMILES will be generated from the MOL file.
    '''
    sdf_path = Path(args.sdf)
    work_dir = sdf_path.parent if args.work_dir == '' else Path(args.work_dir)
    if args.smi != '':
        df_smi = read_smi(args.smi)
        df_smi = df_smi.set_index('Name')
    sdf_supply = Chem.SDMolSupplier(args.sdf)
    sdf_writer = Chem.SDWriter(f'{sdf_path.stem}_prepared.sdf')
    process_bar = ShowProcess(len(sdf_supply), 'Done!')
    pose_count_dict = {}
    for i, mol in enumerate(sdf_supply):
        if mol.GetNumAtoms() > 1000:
            continue  # A simple way to ignore the protein.
        mol_name = mol.GetProp('_Name')
        if mol_name in pose_count_dict.keys():
            pose_count_dict[mol_name] += 1
        else:
            pose_count_dict[mol_name] = 0
        # mol.SetProp('_Name', f'{mol_name}_{pose_count_dict[mol_name]}')  #The name does not include the
        mol.SetProp('Docking_score', mol.GetProp('r_i_docking_score'))
        mol.SetProp('Pose_id', str(pose_count_dict[mol_name]))
        if args.smi != '' and mol_name in df_smi.index:
            mol.SetProp('SMILES', df_smi.loc[mol_name]['SMILES'])
        else:
            smi = Chem.MolToSmiles(mol)
            mol.SetProp('SMILES', smi)
        sdf_writer.write(mol)
        process_bar.show_process()
        # if i > 5: break # For debug
    df_posecount = pd.DataFrame(list(pose_count_dict.items()),
                                columns=['Ligand_name', 'Pose_count'])
    df_posecount.to_csv(f'{work_dir/"pose_count.csv"}')


    # sys.exit()
def main(args):
    prepare_sdf_glide(args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smi", help="csv file of SMILES", default='')
    parser.add_argument("--sdf",
                        help="SDF file of the glide docking result.",
                        default='')
    parser.add_argument("--work_dir",
                        help="The directory to save the results.",
                        default='')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)