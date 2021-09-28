try:
    from openbabel import pybel
except:
    import pybel
import pandas as pd
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm
import os
from pathlib import Path
import rdkit
from rdkit import Chem
from pebble import concurrent, ProcessPool
from concurrent.futures import TimeoutError

glide = "/mnt/home/zhangjie/Bin/Schrodinger2017/glide"
structconvert = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/structconvert"
prepwizard = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/prepwizard"


def find_files(path):
    files = os.listdir(path)
    maegz_list = []
    for file in files:
        file_sp = file.split('_')
        if len(file_sp) < 2:
            print(file+'\t was omitted!')
            continue
        elif file_sp[-1] == 'pv.maegz':
            maegz_list.append(file)
    print(f"{len(maegz_list)} files have been detected!")
    return maegz_list


def smi_mae(id_smi):
    ''' The SMILES of ligand will be transformed into maestro format.
    '''
    # chembl_id = col['ChEMBL ID']
    chembl_id = id_smi[0]
    smi = id_smi[1]
    # path=Path(path)
    opfile = f'{chembl_id}'
    print(opfile)
    try:
        # if 1:
        # smi = col['Smiles']
        mol = pybel.readstring("smi", smi)
        # strip salt
        mol.OBMol.StripSalts(10)
        mols = mol.OBMol.Separate()
        # print(pybel.Molecule(mols))
        mol = pybel.Molecule(mols[0])
        for imol in mols:
            imol = pybel.Molecule(imol)
            if len(imol.atoms) > len(mol.atoms):
                mol = imol

        # print(mol)
        mol.addh()
        mol.title = chembl_id
        # print(mol)
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='mol2', filename=f'{opfile}.mol2', overwrite=True)
        os.system(f'{structconvert} -imol2 {opfile}.mol2 -omae {opfile}.mae')
        #  clean
        if os.path.exists(f'{opfile}.mol2'):
            os.system(f'rm {opfile}.mol2')
    except:
        print(f"Tranformation of {smi} failed! ")
        if os.path.exists(f'{opfile}.mol2'):
            os.system(f'rm {opfile}.mol2')
        return 0


def write_dockInput(id_smi, args):
    dockInput_new_file = f'{id_smi[0]}.in'
    dockInput_new_f = open(dockInput_new_file, 'w')
    with open(args.dockInput_template, 'r') as dockInput_template_f:
        for line in dockInput_template_f.readlines():
            line_new = line.replace('$MaeFile', f'{str(id_smi[0])}.mae')
            # line_new = line_new.replace('$n_jobs', str(n_jobs))
            dockInput_new_f.write(line_new)
    dockInput_template_f.close()
    dockInput_new_f.close()
    return dockInput_new_file


def dock(id_smi, args):
    # dock a single compounds
    smi_mae(id_smi)
    dockInput_new_file = write_dockInput(id_smi, args)
    print(f'dockInput_new_f= {dockInput_new_file}')
    os.system(f'{glide} -WAIT -OVERWRITE  -NOJOBID  {dockInput_new_file}')
    # clean the output
    tempFiles = [f"{id_smi[0]}.in", f"{id_smi[0]}.mae"]
    for ifile in tempFiles:
        os.system(f'rm {ifile}')


def main(args):
    input_df = pd.read_csv(args.dataset)
    os.system(f'mkdir {args.save_path}')
    if args.subset != '':
        save_path = f"{args.save_path}/{args.subset}"
        low_idx = int(args.subset)*100000
        up_idx = (int(args.subset)+1) * 100000
    else:
        low_idx = -1
        up_idx = len(input_df)+1000
        save_path = args.save_path
    os.system(f'mkdir {save_path}')
    os.chdir(f"{save_path}")
    # Load SMILES section
    print('#'*10+'\tLoading SMILES start!\t'+'#'*10)
    IdSmi_list = []
    docked_files = find_files('./')
    Skipped_count = 0
    for idx, col in input_df.iterrows():
        if idx <= up_idx and idx >= low_idx:
            print(f"Loading SMILES: {idx} {col['Smiles']}")
            try:
                # if 1:
                ChEMBL_ID = col['ChEMBL ID']
                mol = Chem.MolFromSmiles(col['Smiles'])
                atoms = mol.GetAtoms()
                natm = len(atoms)
                natm = 10
                if f'{ChEMBL_ID}_pv.maegz' in docked_files:
                    print(
                        f"The ligand has been docked before!\n{ChEMBL_ID}_pv.maegz")
                    continue
                # elif natm > 50:
                #     Skipped_count += 1
                #     print(
                #         f"The ligand is too huge and will be skipped! Skipped count: {Skipped_count} ")
                #     continue
                else:
                    IdSmi_list.append([col['ChEMBL ID'], col['Smiles']])
            except Exception as e:
                print(e)
        if idx > up_idx:
            break
    print('#'*10+'\tDocking start!\t'+'#'*10)
    # with ProcessPool(max_workers=args.n_jobs) as pool:
    #     print("RUNING POOL!!!!")
    #     for ismiId in IdSmi_list:
    #         future = pool.schedule(dock, args=[ismiId], timeout=600)
    with Pool(args.n_jobs) as pool:
        dock_p = partial(dock, args=args)
        res_list = [x for x in tqdm(
            pool.imap(dock_p, list(IdSmi_list)),
            total=len(IdSmi_list),
            miniters=100
        )
            if x is not None]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dockInput_template", help="The template file of the glide dock input file. The absolute path is necessary!", type=str,
                        default='')
    parser.add_argument("--save_path", help="The directory to save the docking results.", type=str,
                        default='')
    parser.add_argument(
        "--dataset", help="ligand dataset (SMILES in csv)", default='')
    parser.add_argument(
        "--sdf", help="sdf file of molecules", default='')
    parser.add_argument("--subset", type=str,
                        help="subset subset of dataset (100,000)",
                        default='')
    parser.add_argument("--n_jobs", type=int,
                        help="cpu cores", default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # test section

    args = get_parser()
    main(args)
