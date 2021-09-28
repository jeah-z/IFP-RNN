from __future__ import print_function

import sys
import os
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
try:
    from openbabel import pybel
except ImportError:
    import pybel
import pandas as pd
import argparse
import glob
from rdkit import Chem


def path_to_filename(path_input):
    name = os.path.basename(path_input)
    name = name.replace('_', '.').split('.')
    file_name = name[0]
    return file_name


def find_files(path, suffix):
    files = os.listdir(path)
    pdbqt_list = []
    for file in files:
        print(f"Processing {file}!")
        file_sp = file.split('_')
        if len(file_sp) < 2:
            print(file+'\t was omitted!')
            continue
        elif file_sp[-1] == suffix:
            pdbqt_list.append(file)
    print(f"{len(pdbqt_list)} files have been detected!")
    return pdbqt_list


def parse_ligand_vina(ligand_file):
    ligand_name = path_to_filename(ligand_file)
    # read the scorelist in the pdbqt file
    scorelist = []
    try:
        ligand_out_lines = open(ligand_file, 'r')
    except FileNotFoundError:
        print("Ligand output file: '%s' can not be found" % (ligand_file))
        sys.exit(1)
    for line in ligand_out_lines:
        line = line.split()
        if len(line) > 2:
            if line[2] == "RESULT:":
                scorelist.append(line[3])
    # process docking pose of ligands after docking
    convert = ob.OBConversion()
    convert.SetInFormat("pdbqt")
    ligands = ob.OBMol()
    docked_ligands = []
    not_at_end = convert.ReadFile(ligands, ligand_file)
    while not_at_end:
        docked_ligands.append(ligands)
        ligands = ob.OBMol()
        not_at_end = convert.Read(ligands)

    docking_results = {
        'docked_ligands': docked_ligands,
        'scorelist': scorelist
    }
    return docking_results


def write_sdf(ligand_list, sdf_file, ligand_df, seed_smi=''):
    # sdFile = pybel.Outputfile('sdf', sdf_file, overwrite=True)
    sdw = Chem.SDWriter(sdf_file)
    index = 0
    for iligand in ligand_list:
        print(f'Processing {iligand}!')
        # try:
        if 1:
            # ligand_name = path_to_filename(iligand)
            ligand_path = iligand[0]
            ligand_name = iligand[1]
            ligand_pose = int(iligand[2])
            # seed_smi = iligand[3]
            docking_result = parse_ligand_vina(ligand_path)
            mol = pybel.Molecule(docking_result['docked_ligands'][ligand_pose])
            mol.title = f'{ligand_name}_{ligand_pose}'
            mol.write('mol', f"{ligand_name}_{ligand_pose}.mol")
            try:
                rd_mol = Chem.MolFromMolFile(
                    f"{ligand_name}_{ligand_pose}.mol")
                os.system(f"rm {ligand_name}_{ligand_pose}.mol")
            except Exception as e:
                print(e)
            if rd_mol == None:
                continue
            # mol = Chem.MolFromMolBlock(mol)
            # mol.title = f'{ligand_name}_{ligand_pose}'
            print(rd_mol)
            # break
            if index > 0 and rd_mol != None:

                # mol.ifpSim = ligand_df.loc[f'{ligand_name}_pose_{ligand_pose}', "ifpSim"]
                ifpSim = ligand_df.loc[f'{ligand_name}_{ligand_pose}', "ifpSim"]
                rd_mol.SetProp(
                    "ifpSim", str(ifpSim))
                # mol.bitRec = ligand_df.loc[f'{ligand_name}_pose_{ligand_pose}', "bitRec"]
                bitRec = ligand_df.loc[f'{ligand_name}_{ligand_pose}', "bitRec"]
                rd_mol.SetProp(
                    "bitRec", str(bitRec))
                # mol.dockScore = ligand_df.loc[f'{ligand_name}_pose_{ligand_pose}', "dockScore"]
                dscore = ligand_df.loc[f'{ligand_name}_{ligand_pose}', "dockScore"]
                rd_mol.SetProp("dockScore", str(dscore))
                # mol.smi = ligand_df.loc[f'{ligand_name}_pose_{ligand_pose}', "smi"]
                smi = ligand_df.loc[f'{ligand_name}_{ligand_pose}', "smi"]
                rd_mol.SetProp("SMILES", str(smi))
                rd_mol.SetProp("seed_smi", str(seed_smi))
                # mol.molSim = ligand_df.loc[f'{ligand_name}_pose_{ligand_pose}', "molSim"]
                molsim = ligand_df.loc[f'{ligand_name}_{ligand_pose}', "molSim"]
                rd_mol.SetProp("molSim", str(molsim))
                # mol.IFP_Score = ligand_df.loc[f'{ligand_name}_pose_{ligand_pose}', "IFP_Score"]
                ifpscore = ligand_df.loc[f'{ligand_name}_{ligand_pose}', "IFP_Score"]
                rd_mol.SetProp("IFP_Score", str(ifpscore))
            index += 1
            # sdFile.write(mol)
            sdw.write(rd_mol)
        # except Exception as e:
        #     print(e)
        #     continue


def pdbqt_sdf(pdbqt):
    print(f"Transferring the {pdbqt} to sdf!")
    ligand_name = os.path.basename(pdbqt)
    ligand_name = ligand_name.split('.')[0]
    sdFile = pybel.Outputfile('sdf', f"{ligand_name}.sdf", overwrite=True)
    docking_result = parse_ligand_vina(pdbqt)
    score_list = []
    for idx, val in enumerate(docking_result['docked_ligands']):
        mol = pybel.Molecule(val)
        mol.title = f'{ligand_name}_{idx}'
        mol.docking_score = docking_result['scorelist'][idx]
        score_list.append(
            [f'{ligand_name}_{idx}', docking_result['scorelist'][idx]])
        sdFile.write(mol)
    sdFile.close()
    return score_list


def main(args):
    folder_transfer = 0
    # A previous version
    if 1:
        dfolder = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Results/dockTmp_dScorePP'
        sorted_folder = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Results/sorted_IFPscore'
        seed_dfolder = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Data/a2a_active_dock/0_pdbqt'
        seed_smi = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Data/a2a_chembl.csv'
        seed_list = find_files(sorted_folder, 'sorted.csv')

        df_seedSmi = pd.read_csv(seed_smi)
        df_seedSmi = df_seedSmi.set_index('ChEMBL ID')

        for iseed in seed_list:
            print(f"Processing seed: {iseed} !")
            seed_name = iseed.replace("-", "_").strip().split("_")[0]
            seed_smi = df_seedSmi.loc[seed_name, 'Smiles']
            ipose = iseed.replace("-", "_").strip().split("_")[1]
            seed_file = f"{seed_dfolder}/{seed_name}_out.pdbqt"
            ligands = [[seed_file, "seed", ipose]]
            ligand_df = pd.read_csv(f"{sorted_folder}/{iseed}")
            # print(f"ligand_df={ligand_df}")
            ligand_df["index"] = ligand_df["Molecule"]
            ligand_df.set_index('index', inplace=True)
            ligand_top100 = ligand_df['Molecule'][:200]
            for ligand in ligand_top100:
                ligand_sp = ligand.split("_")
                ligand_name = ''
                for i in range(len(ligand_sp)-1):
                    if ligand_name == '':
                        ligand_name = ligand_sp[i]
                    else:
                        ligand_name += f"_{ligand_sp[i]}"
                ligand_pose = ligand_sp[-1]
                ligands.append(
                    [f"{dfolder}/{seed_name}_{ipose}_1.0/glide_pdbqt/{ligand_name}_out.pdbqt", ligand_name, ligand_pose])

            print(f'ligands={ligands}')
            # ligands = [f'{folder}/{l}_out.pdbqt' for l in ligands]
            write_sdf(
                ligands, f"{sorted_folder}/cdk2_{seed_name}.sdf", ligand_df, seed_smi)
    # A simple function to convert the all the pdbqts to sdf in the target folder

    if folder_transfer:
        os.chdir(args.pdbqt_folder)
        pdbqts = glob.glob('*_out.pdbqt')
        score_list = []
        for pdbqt in pdbqts:
            dscore = pdbqt_sdf(pdbqt)
            score_list.append(dscore[0])
        score_df = pd.DataFrame(score_list, columns=["Name", "dScore"])
        score_df = score_df.sort_values(
            ["dScore"], ascending=False)
        score_df.to_csv("docking_score.csv", index=False)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdbqt_folder", help="the folder of pdbqts",
                        default='')
    # parser.add_argument("--save_dir", help="adirectory to save the pdbs",
    #                     default='./Results_crystal/aligned_pdbs')
    # parser.add_argument("--target_pdb", help="abosolute path of the target pdb for aligning",
    #                     type=str, default='/mnt/home/zhangjie/Projects/cRNN/AIFP/Data/Protein/protein.pdb')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)
