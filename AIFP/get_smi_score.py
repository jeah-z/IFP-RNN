from model.toolkits.parse_conf import parse_config_vina, parse_protein_vina, parse_ligand_vina
import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
try:
    from openbabel import pybel
except:
    import pybel
# from metrics_utils import logP, QED, SA, weight, NP
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm


def walk_folder(path, suffix):
    # processed = []
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


def prepare_ecfp(args):
    dataset = args.dataset
    path = args.path
    df = pd.read_csv(dataset)
    # smi_df = pd.read_csv(args.smi)
    # smi_df = smi_df.set_index('ChEMBL ID')
    df['index'] = df['Molecule']
    df = df.set_index('index')
    # counts = 0
    # for index in df.index:
    #     smi_index = index.strip().split("_")[0]
    #     counts += 1
    #     try:
    #         # print(smi_df.loc[smi_index, 'Smiles'])
    #         smiRaw = smi_df.loc[smi_index, 'Smiles']
    #         mol = pybel.readstring("smi", smiRaw)
    #         # strip salt
    #         mol.OBMol.StripSalts(10)
    #         mols = mol.OBMol.Separate()
    #         # print(pybel.Molecule(mols))
    #         mol = pybel.Molecule(mols[0])
    #         for imol in mols:
    #             imol = pybel.Molecule(imol)
    #             if len(imol.atoms) > len(mol.atoms):
    #                 mol = imol
    #         smi_clean = mol.write('smi')
    #         smi_clean = smi_clean.replace('\n', '')
    #         smi_clean = smi_clean.split()[0]
    #         df.loc[index, 'smi'] = smi_clean
    #         print(f'NO.{counts}: {smi_clean} was processed successfully')
    #     except Exception as e:
    #         print(e)
    #         continue
    df = df.dropna(axis=0, how='any')
    smiList = df['smi']
    index = df['Molecule']
    # print(smiList)
    new_index, ecfpList = [], []
    for i in range(len(index)):
        try:
            smi = smiList[i]
            if i % 1000 == 0:
                print(f"index: {i}; smi= {smi}")
            mol = Chem.MolFromSmiles(smi)
            ecfp = AllChem.GetMorganFingerprintAsBitVect(
                mol, 3, nBits=1024)
            ecfp=[index[i]]+list(ecfp)
            ecfpList.append(ecfp)
            # new_index.append()
        except:
            continue

        # molList = [Chem.MolFromSmiles(smi)
        #         for smi in smiList]
        # ecfpList = [list(AllChem.GetMorganFingerprintAsBitVect(
        #     mol, 3, nBits=1024)) for mol in molList]
    # print(ecfpList)
    colName = ['index']+[f'ecfp{i}' for i in range(len(ecfpList[0])-1)]
    # print(colName)
    dfEcfp = pd.DataFrame(ecfpList, columns=colName)
    # dfEcfp['index'] = new_index
    dfEcfp = dfEcfp.set_index('index')
    # print(dfEcfp)
    # print(df)
    dfEcfp = pd.concat([df, dfEcfp], axis=1)
    dfEcfp = dfEcfp.dropna(axis=0, how='any')
    suffix = '_ecfpSmi.csv'
    outdf = dataset.replace('.csv', suffix)
    # dfEcfp = dfEcfp.dropna(axis=0, how='any')
    # if not os.path.exists(outdf):
    dfEcfp.to_csv(outdf, index=False)


def prepare_DScorePPropIFP(args, getScore=True, getSMILES=True):
    dataset = args.dataset
    # smiDataset = args.smi
    df = pd.read_csv(dataset)
    df = df.set_index('Molecule')
    smi_df = pd.read_csv(args.smi)
    smi_df = smi_df.set_index('ChEMBL ID')
    path = args.path

    # index = df.index
    counts = 0
    if getSMILES:
        for index in df.index:
            smi_index = index.strip().split("_")[0]
            try:
                counts += 1
                # print(smi_df.loc[smi_index, 'Smiles'])
                smiRaw = smi_df.loc[smi_index, 'Smiles']
                mol = pybel.readstring("smi", smiRaw)
                # strip salt
                mol.OBMol.StripSalts(10)
                mols = mol.OBMol.Separate()
                # print(pybel.Molecule(mols))
                mol = pybel.Molecule(mols[0])
                for imol in mols:
                    imol = pybel.Molecule(imol)
                    if len(imol.atoms) > len(mol.atoms):
                        mol = imol
                smi_clean = mol.write('smi')
                smi_clean = smi_clean.replace('\n', '')
                smi_clean = smi_clean.split()[0]
                df.loc[index, 'smi'] = smi_clean
                print(f'NO.{counts}: {smi_clean} was processed successfully')
            except Exception as e:
                print(e)
                continue
    # df = df.dropna(axis=0, how='any')
    if getScore:
        files = walk_folder(path, '_out.pdbqt')
        count = 0
        for file in files:
            count += 1
            # if count > 10:
            #     break
            print(f'count: {count}')
            try:
                # if 1:
                outfile = os.path.join(path, file)
                ligand_dic = parse_ligand_vina(outfile)
                score = ligand_dic['scorelist']
                filename = file.replace('_out.pdbqt', '')
                cal_switch = 0
                for pose_idx in range(5):
                    df.loc[f'{filename}_{pose_idx}',
                           'score_0'] = score[pose_idx]
                    smi = df.loc[f'{filename}_{pose_idx}', 'smi']
                    print(smi)
                    if cal_switch < 1:
                        mol = Chem.MolFromSmiles(smi)
                        logp = Descriptors.MolLogP(mol)
                        tpsa = Descriptors.TPSA(mol)
                        molwt = Descriptors.ExactMolWt(mol)
                        hba = rdMolDescriptors.CalcNumHBA(mol)
                        hbd = rdMolDescriptors.CalcNumHBD(mol)
                        qed = QED.qed(mol)
                        cal_switch = 3

                    df.loc[f'{filename}_{pose_idx}', 'logP'] = logp
                    df.loc[f'{filename}_{pose_idx}', 'TPSA'] = tpsa
                    df.loc[f'{filename}_{pose_idx}', 'MW'] = molwt
                    df.loc[f'{filename}_{pose_idx}', 'HBA'] = hba
                    df.loc[f'{filename}_{pose_idx}', 'HBD'] = hbd
                    df.loc[f'{filename}_{pose_idx}', 'QED'] = qed
            # logp = logP(mol)
            # df.loc[filename, 'logP'] = logp
            # qed = QED(mol)
            # df.loc[filename, 'QED'] = qed
            # sa = SA(mol)
            # df.loc[filename, 'SA'] = sa
            # wt = weight(mol)
            # df.loc[filename, 'Wt'] = wt
            # np = NP(mol)
            # df.loc[filename, 'NP'] = np
            except Exception as e:
                print(e)
                continue
    suffix = '_dScorePP.csv'
    # df = df.sort_values(by='score_0', ascending=True)
    outdf = dataset.replace('.csv', suffix)
    df = df.dropna(axis=0, how='any')
    # df['score_0'] = df['score_0'].astype(float)
    # if not os.path.exists(outdf):
    df.to_csv(outdf, index=True)


def prepare_IFPsmi(args, getScore=False, getSMILES=True):
    dataset = args.dataset
    # smiDataset = args.smi
    df = pd.read_csv(dataset)
    df = df.set_index('Molecule')
    smi_df = pd.read_csv(args.smi)
    smi_df = smi_df.set_index('ChEMBL ID')
    path = args.path

    # index = df.index
    counts = 0
    if getSMILES:
        for index in df.index:
            smi_index = index.strip().split("_")[0]
            try:
                counts += 1
                print(smi_df.loc[smi_index, 'Smiles'])
                smiRaw = smi_df.loc[smi_index, 'Smiles']
                mol = pybel.readstring("smi", smiRaw)
                # strip salt
                mol.OBMol.StripSalts(10)
                mols = mol.OBMol.Separate()
                # print(pybel.Molecule(mols))
                mol = pybel.Molecule(mols[0])
                for imol in mols:
                    imol = pybel.Molecule(imol)
                    if len(imol.atoms) > len(mol.atoms):
                        mol = imol
                smi_clean = mol.write('smi')
                smi_clean = smi_clean.replace('\n', '')
                smi_clean = smi_clean.split()[0]
                df.loc[index, 'smi'] = smi_clean
                print(f'NO.{counts}: {smi_clean} was processed successfully')
            except Exception as e:
                print(e)
                continue
    # df = df.dropna(axis=0, how='any')
    suffix = '_AIFPsmi.csv'
    outdf = dataset.replace('.csv', suffix)
    df = df.dropna(axis=0, how='any')
    # if not os.path.exists(outdf):
    df.to_csv(outdf, index=True)


def main(args):
    if args.model == 'ecfp':
        prepare_ecfp(args)
    elif args.model == 'dScorePP':
        prepare_DScorePPropIFP(args)
    elif args.model == 'aifp':
        prepare_IFPsmi(args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="docking results path",
                        default='')
    # parser.add_argument("--save", help="dataframe name of IFP",
    #                     default='IFP')
    parser.add_argument("--dataset", help="IFP dataset path", type=str,
                        default='')
    parser.add_argument("--smi", help="SMILES dataset path", type=str,
                        default='./Data/ChEMBL27.csv')
    parser.add_argument("--model", help="model name: aifp,ecfp or dScorePP", type=str,
                        default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    # config = parse_config_vina(args.config)
    main(args)
