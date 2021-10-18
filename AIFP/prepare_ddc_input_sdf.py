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
from pandarallel import pandarallel


def cal_ecfp(smi):
    try:
        smi = list(smi)[0]
        print(smi)
        mol = Chem.MolFromSmiles(smi)
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
        return list(ecfp)
    except Exception as e:
        print(e)
        return []


def prepare_ecfp(args):
    df_IFP = pd.read_csv(args.dataset).set_index('Molecule', drop=False)
    df_info = pd.read_csv(args.info).set_index('Name', drop=False)

    ##  Get SMILES
    pandarallel.initialize()
    df_IFP['smi'] = df_IFP.parallel_apply(
        lambda x: df_info.loc[x['Molecule'], 'SMILES'], axis=1)
    ## Calculate ECFP fingerprint
    new_columns = [f'ecfp{i}' for i in range(1024)]
    df_IFP[new_columns] = df_IFP.parallel_apply(
        lambda x: cal_ecfp(df_info.loc[x['Molecule'], ['SMILES']]),
        axis=1,
        result_type='expand')
    suffix = '_ecfpSmi.csv'
    # df = df.sort_values(by='score_0', ascending=True)
    outdf = args.dataset.replace('.csv', suffix)
    df_IFP = df_IFP.dropna(axis=0, how='any')
    # df['score_0'] = df['score_0'].astype(float)
    # if not os.path.exists(outdf):
    df_IFP.to_csv(outdf, index=False)


def cal_props(smi):
    try:
        smi = list(smi)[0]
        mol = Chem.MolFromSmiles(smi)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        molwt = Descriptors.ExactMolWt(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        qed = QED.qed(mol)
        return logp, tpsa, molwt, hba, hbd, qed  #['logP', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']
    except Exception as e:
        print(e)
        return []


def prepare_DScorePPropIFP(args, getScore=True, getSMILES=True):
    df_IFP = pd.read_csv(args.dataset).set_index('Molecule', drop=False)
    df_info = pd.read_csv(args.info).set_index('Name', drop=False)

    ##  Get SMILES
    pandarallel.initialize()
    df_IFP[['smi', 'score_0']] = df_IFP.parallel_apply(
        lambda x: df_info.loc[x['Molecule'], ['SMILES', 'Docking_score']],
        axis=1)

    ## Calculate physical properties
    new_columns = ['logP', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']
    df_IFP[new_columns] = df_IFP.parallel_apply(
        lambda x: cal_props(df_info.loc[x['Molecule'], ['SMILES']]),
        axis=1,
        result_type='expand')
    suffix = '_dScorePP.csv'
    # df = df.sort_values(by='score_0', ascending=True)
    outdf = args.dataset.replace('.csv', suffix)
    df_IFP = df_IFP.dropna(axis=0, how='any')
    # df['score_0'] = df['score_0'].astype(float)
    # if not os.path.exists(outdf):
    df_IFP.to_csv(outdf, index=False)


def prepare_IFPsmi(args, getScore=False, getSMILES=True):
    df_IFP = pd.read_csv(args.dataset).set_index('Molecule', drop=False)
    df_info = pd.read_csv(args.info).set_index('Name', drop=False)

    ##  Get SMILES
    df_IFP['smi'] = df_IFP.apply(
        lambda x: df_info.loc[x['Molecule'], 'SMILES'], axis=1)
    suffix = '_AIFPsmi.csv'
    outdf = args.dataset.replace('.csv', suffix)
    df_IFP = df_IFP.dropna(axis=0, how='any')
    # if not os.path.exists(outdf):
    df_IFP.to_csv(outdf, index=None)


def main(args):
    if args.type == 'ecfp':
        prepare_ecfp(args)
    elif args.type == 'dScorePP':
        prepare_DScorePPropIFP(args)
    elif args.type == 'aifp':
        prepare_IFPsmi(args)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--info",
        help=
        "File path of ligands information, including docking score, SMILES etc.",
        default='')
    parser.add_argument("--dataset",
                        help="IFP dataset path",
                        type=str,
                        default='')
    parser.add_argument("--type",
                        help="model type: aifp,ecfp or dScorePP",
                        type=str,
                        default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    # config = parse_config_vina(args.config)
    main(args)
