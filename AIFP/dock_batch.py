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


def smi_pdb(id_smi):
    # chembl_id = col['ChEMBL ID']
    chembl_id = id_smi[0]
    smi = id_smi[1]
    # path=Path(path)
    opfile = f'{chembl_id}.pdb'
    print(opfile)
    try:
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
        # print(mol)
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='pdb', filename=str(opfile), overwrite=True)
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        return 0


def prepare_ligand(file, args):
    prepare_ligand4 = f'{args.mgltools}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'
    # SavePath=Path(save_path)
    file = Path(file)
    op_file = f'{file.stem}.pdbqt'
    cmd = f"{prepare_ligand4} -l {file} -o {op_file} -A hydrogens"
    try:
        os.system(cmd)
        os.system(f'rm {file}')
        print(f"{file} has been processed successfully! ")
        return 1
    except:
        print(f"{file} has been omitted! ")
        if not os.path.isfile('./docking_failed/'):
            os.system(f'mkdir ./docking_failed/')
        os.system(f'mv {file} ./docking_failed/')
        return 0


def config_create(template, ligand, machine):
    ligand = Path(ligand)
    #job_id = random.randint(1, 999999)
    template_file = open(template, 'r')
    new_config = f'./config_run_{machine}_{ligand.stem}.txt'
    config_file = open(new_config, 'w')
    for line in template_file.readlines():
        line_new = line.replace('$ligand', str(ligand))
        # line_new = line_new.replace('$n_jobs', str(n_jobs))
        config_file.write(line_new)
    return new_config


def run_dock(file_path, save_path, machine,args):
    vina = f'{args.mgltools}/bin/vina'
    if os.path.isfile(file_path):
        os.system(f"mv {file_path} {save_path}/")
        filename = os.path.basename(file_path)
        config_template = args.config_vina
        new_config = config_create(config_template,
                                   f'{save_path}/{filename}', machine)
        cmd = f"{vina} --config {new_config} --log /dev/null"
        print(cmd)
        os.system(cmd)
        os.system(f'rm {new_config}')
        os.system(f'rm {save_path}/{filename}')
        print(f"{filename} has been processed successfully! ")
        return 1
    else:
        return 0


def dock(id_smi, save_path='./', machine='LocalPC',args=''):
    try:
        smi_pdb(id_smi)
        file = f'{id_smi[0]}.pdb'
        prepare_ligand(file,args)
        file = f'{id_smi[0]}.pdbqt'
        print(f'procesing: {id_smi} ')
        print("stoped here 1")
        run_dock(file, save_path, machine,args)
        print("stoped here 2")
        # os.system(f'rm {file}')
        return 1
    except Exception as e:
        print(f"Something went wrong while processing: {id_smi}")
        print(e)
        return 0


def find_files(path):
    files = os.listdir(path)
    pdbqt_list = []
    for file in files:
        file_sp = file.split('_')
        if len(file_sp) < 2:
            print(file+'\t was omitted!')
            continue
        elif file_sp[-1] == 'out.pdbqt':
            pdbqt_list.append(file)
    print(f"{len(pdbqt_list)} files have been detected!")
    return pdbqt_list


def main(args):
    # parameters
    dataset = args.dataset
    n_jobs = args.n_jobs
    subset = args.subset
    save_path = args.save_path
    os.system(f'mkdir {save_path}')
    save_path = Path(save_path)
    input_pd = pd.read_csv(dataset)
    if subset != '':
        save_path = save_path/subset
        os.system(f'mkdir {save_path}')
        low_idx = (int(subset)-1)*100000
        up_idx = int(subset) * 100000
    else:
        low_idx = -1
        up_idx = len(input_pd)+1000
    # Load SMILES
    print('#'*10+'\tLoading SMILES start!\t'+'#'*10)

    IdSmi_list = []

    docked_files = find_files(str(save_path))
    Skipped_count = 0
    for idx, col in input_pd.iterrows():
        if idx <= up_idx and idx >= low_idx:
            print(f"Loading SMILES: {idx} {col['Smiles']}")
            try:
                ChEMBL_ID = col['ChEMBL ID']
                mol = Chem.MolFromSmiles(col['Smiles'])
                atoms = mol.GetAtoms()
                natm = len(atoms)
                natm = 10
                if f'{ChEMBL_ID}_out.pdbqt' in docked_files:
                    print(
                        f"The ligand has been docked before!\n{ChEMBL_ID}_out.pdbqt")
                    continue
                elif natm > 50:
                    Skipped_count += 1
                    print(
                        f"The ligand is too huge and will be skipped! Skipped count: {Skipped_count} ")
                    continue
                else:
                    IdSmi_list.append([col['ChEMBL ID'], col['Smiles']])
            except Exception as e:
                print(e)
        if idx > up_idx:
            break
    print('#'*10+'\tLoading SMILES finished!\t'+'#'*10)
    # dock
    print('#'*10+'\tDocking start!\t'+'#'*10)

    with ProcessPool(max_workers=n_jobs) as pool:
        print("RUNING POOL!!!!")
        for ismiId in IdSmi_list:
            future = pool.schedule(dock, args=[ismiId, str(
                save_path), args.machine, args], timeout=600)
    # with ProcessPool(max_workers=n_jobs) as pool:
    #     dock_p = partial(dock, save_path=str(save_path),
    #                      machine=args.machine)
    #     future = pool.map(dock_p, IdSmi_list, timeout=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", help="ligand dataset (SMILES in csv)", default='')
    parser.add_argument(
        "--save_path", help="path to save docking results", default='')
    parser.add_argument("--n_jobs", type=int,
                        help="cpu cores", default=1)
    parser.add_argument("--machine", type=str,
                        help="machine name", default='47')
    parser.add_argument("--subset", type=str,
                        help="subset subset of dataset (100,000)",
                        default='')
    parser.add_argument("--mgltools", type=str,
                        help="path of mgltools",
                        default='./MGLTools-1.5.7')
    parser.add_argument("--config_vina", type=str,
                        help="path of config_vina.txt",
                        default='./config_vina.txt')
    args = parser.parse_args()
    main(args)
