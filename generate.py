from __future__ import print_function
# import ddc.ddc_pub.ddc_v3 as ddc
import pandas as pd
import argparse
import numpy as np
import pickle
import rdkit
from rdkit import Chem
from ddc.ddc_pub import ddc_v3 as ddc
import numpy as np
import rdkit
from rdkit import DataStructs
import os
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path

# import h5py
# import ast
import pickle


def save_obj(obj, name):
    # os.system('mkdir obj')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)


def add_random(bits):
    bits = np.array(bits)
    rands = np.random.rand(len(bits))/10
    bits = bits+rands
    bits[bits < 0.5] = 0
    return bits


def cal_valid(smiList):
    total = len(smiList)
    valid = 0
    valSmis = []
    for smi in smiList:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid += 1
            valSmis.append(smi)
    valid_rate = valid/float(total+0.01)*100
    return valid_rate, valSmis


def plot_valid(df):

    sns.set(style='ticks')
    plt.figure(figsize=(7, 4.8))
    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    ifg = df
    paper_rc = {'lines.linewidth': 2, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    sns.lineplot(x='epoch', y='validity', data=ifg)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    # plt.xlim(0, 600)
    plt.ylabel('Validity', fontsize=14)
    # logPath = Path(args.log)
    # logName = logPath.name
    title = 'Validity of SMILES'
    plt.title(title)
    plt.savefig(
        os.path.join('images', title+'.pdf')
    )
    plt.savefig(
        os.path.join('images', title+'.png'),
        dpi=250
    )


def prepare_input(ifp_df, seedDf, job_type=''):
    ifp_df['index'] = ifp_df['Molecule']
    ifp_df = ifp_df.set_index('index')
    if job_type == 'all_poses':
        '''For the situation that poses are not given!'''
        seedList = []
        mol_names = list(seedDf['Molecule'])
        poses = list(seedDf['Pose'])
        for i in range(len(mol_names)):
            seedList.append(f"{mol_names[i]}_{poses[i]}")
    else:
        '''Pose id is included in the names!'''
        seedList = list(seedDf['Molecule'])

    print(f'Number of seeds: {len(seedList)}')
    print(f'seed molecules: {seedList}')
    dfNew = ifp_df.loc[seedList]
    inputList = []
    colDrop = ['index', 'smi', 'Molecule', 'logP', 'QED', 'SA',
               'Wt', 'NP', 'score_0', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']
    for i in range(1024):
        colDrop.append(f'ecfp{i}')
    for idx, row in dfNew.iterrows():
        smi = row['smi']
        molID = row['Molecule']
        IFP = row.copy()
        row = row.drop(['smi', 'Molecule'])
        '''Get a clean IFP without other informations!'''
        for colName in colDrop:
            try:
                IFP = IFP.drop([colName])
            except Exception as e:
                print(e)
                continue
        row = np.array(row)
        IFP = np.array(IFP)
        # row=add_random(row)
        inputDic = {'smi': smi, 'molID': molID, 'row': row, 'IFP': IFP}
        inputList.append(inputDic)
        print(f'smi:{smi} molID:{molID} row:{row}')
    return inputList


def write_list(listname, op):
    op.writelines('\tIFP: [')
    for itm in listname:
        op.writelines(f'{itm} ')
    op.writelines('] \n')


def cal_similarity(seed, smis):
    seedMol = Chem.MolFromSmiles(seedSmi)
    mols = [Chem.MolFromSmiles(smi) for smi in smis]
    seedFP = AllChem.GetMorganFingerprintAsBitVect(
        seedMol, 3, nBits=1024)
    FPs = [AllChem.GetMorganFingerprintAsBitVect(
        mol, 3, nBits=1024) for mol in mols]


def benchmark_model(args, tempList):
    IFP_Df = pd.read_csv(args.IFP)
    seedDf = pd.read_csv(args.seed)
    inputList = prepare_input(IFP_Df, seedDf, job_type='dscorepp')
    # os.system(f'mkdir {args.save}')
    # validList = []
    model_name = args.model
    colname = []
    # reference = load_obj(f'./AIFP/obj/{ifpRefer[0]}')
    # for iatm in reference:
    #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
    #         colname.append(f'{iatm}_{iifp}')
    # referenceNonAll0 = load_obj(f'./AIFP/obj/{ifpRefer[1]}')
    # NonAll0Idx = [colname.index(itm) for itm in referenceNonAll0]
    # print(f'NonAll0Idx={NonAll0Idx}')
    print(model_name)
    model = ddc.DDC(model_name=model_name)
    # smiDic = []
    for temp in tempList:
        print(f"Sampling at the temperature: {temp}.")
        smiList = []
        for inputDic in inputList:
            seed_smi = inputDic['smi']
            molID = inputDic['molID']
            row = inputDic['row']
            IFP = inputDic['IFP']

            # IFP=np.array(IFP)
            row = np.array([row]*256)
            # print(row)
            print(f'Sampling for molecule: {molID}')
            model.batch_input_length = 256
            # smiles, _ = model.predict_batch(latent=IFP, temp=0.5)
            # print(smiles)
            try:
                smiles = []  # sampling for 20 rounds and 5K smiles
                for isample in range(2):
                    print(f"Sampling for {isample} round!")
                    smi, _ = model.predict_batch(latent=row, temp=temp)
                    smiles.extend(smi)
                smiles = list(smiles)  # remove duplicated smiles
                validity, valSmis = cal_valid(smiles)
                print(
                    f"index: {inputList.index(inputDic)}  validity: {validity}")
                smiList.append({'seedSmi': seed_smi, 'molID': molID, "SeedIFP": list(
                    IFP), 'smis': valSmis, 'validity': validity})
            except Exception as e:
                print(e)
                continue
        save_obj(smiList, f'{args.save}_{temp}')


def sample_model(args, tempList):
    IFP_Df = pd.read_csv(args.IFP)
    seedDf = pd.read_csv(args.seed)
    inputList = prepare_input(IFP_Df, seedDf, random=False)
    # os.system(f'mkdir {args.save}')
    # validList = []
    model_name = args.model
    colname = []
    # reference = load_obj(f'./AIFP/obj/{ifpRefer[0]}')
    # for iatm in reference:
    #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
    #         colname.append(f'{iatm}_{iifp}')
    # referenceNonAll0 = load_obj(f'./AIFP/obj/{ifpRefer[1]}')
    # NonAll0Idx = [colname.index(itm) for itm in referenceNonAll0]
    # print(f'NonAll0Idx={NonAll0Idx}')
    print(model_name)
    model = ddc.DDC(model_name=model_name)
    # smiDic = []
    for temp in tempList:
        smiList = []
        for inputDic in inputList:
            seedSmi = inputDic['smi']
            molID = inputDic['molID']
            row = inputDic['row']
            IFP = inputDic['IFP']

            # IFP=np.array(IFP)
            row = np.array([row]*256)
            # print(row)
            print(f'Sampling for molecule: {molID}')
            model.batch_input_length = 256
            # smiles, _ = model.predict_batch(latent=IFP, temp=0.5)
            # print(smiles)
            try:
                smiles = []  # sampling for 20 rounds and 5K smiles
                for isample in range(20):
                    print(f"Sampling for {isample} round!")
                    smi, _ = model.predict_batch(latent=row, temp=temp)
                    smiles.extend(smi)
                smiles = list(set(smiles))  # remove duplicated smiles
                validity, valSmis = cal_valid(smiles)
                print(
                    f"index: {inputList.index(inputDic)}  validity: {validity}")
                smiList.append({'seedSmi': seedSmi, 'molID': molID, "SeedIFP": list(
                    IFP), 'smis': valSmis, 'validity': validity})
            except Exception as e:
                print(e)
                continue
        save_obj(smiList, f'{args.save}_{args.label}_{temp}')


def benchmark_efcpDrift(args, tempList):
    IFP_Df = pd.read_csv(args.IFP)
    seedDf = pd.read_csv(args.seed)
    inputList = prepare_input(IFP_Df, seedDf, job_type='ecfp')
    # os.system(f'mkdir {args.save}')
    # validList = []
    model_name = args.model
    colname = []
    # reference = load_obj(f'./AIFP/obj/{ifpRefer[0]}')
    # for iatm in reference:
    #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
    #         colname.append(f'{iatm}_{iifp}')
    # referenceNonAll0 = load_obj(f'./AIFP/obj/{ifpRefer[1]}')
    # NonAll0Idx = [colname.index(itm) for itm in referenceNonAll0]
    # print(f'NonAll0Idx={NonAll0Idx}')
    print(model_name)
    model = ddc.DDC(model_name=model_name)
    switch_num = int(args.switch_pt*1024/100)
    print(f'{switch_num}/1024 bits will be switched!')
    # smiDic = []
    for temp in tempList:
        smiList = []
        for inputDic in inputList:
            smi = inputDic['smi']
            molID = inputDic['molID']
            row = inputDic['row']
            print(row)
            IFP = inputDic['IFP']
            row = np.array([row]*256)
            # rows = []
            for i in range(256):
                # irow = row.copy()
                # print(len(irow))
                for jbit in range(0, 1024):
                    switch = np.random.randint(1, 1024)
                    if switch < switch_num:
                        if int(row[i][jbit]) == 0:
                            row[i][jbit] = 1
                        else:
                            row[i][jbit] = 0
            # IFP=np.array(IFP)
            # rows = np.array(rows)
            # print(rows)

            print(f'Sampling for molecule: {molID}')
            model.batch_input_length = 256
            # smiles, _ = model.predict_batch(latent=IFP, temp=0.5)
            # print(smiles)
            try:
                smiles, _ = model.predict_batch(latent=row, temp=temp)
                smiles = list(smiles)
                validity, valSmis = cal_valid(smiles)
                print(
                    f"index: {inputList.index(inputDic)}  validity: {validity}")
                smiList.append({'seedSmi': smi, 'molID': molID, "SeedIFP": list(
                    IFP), 'smis': valSmis, 'validity': validity})
            except Exception as e:
                print(e)
                continue

        os.system(f"mkdir {Path(args.save).parent}")
        save_obj(smiList, f'{args.save}_{args.switch_pt}_{temp}')


def generate_smis(args):
    IFP_Df = pd.read_csv(args.IFP)
    seedDf = pd.read_csv(args.seed)
    inputList = prepare_input(IFP_Df, seedDf, random=True, num=args.num)
    model_name = args.model
    print(model_name)
    model = ddc.DDC(model_name=model_name)
    for temp in [0.1, 0.2, 0.4, 0.6, 1.0]:
        os.system(f'mkdir sampled_smiles/{args.save}')
        opFileName = f'sampled_smiles/{args.save}/temp_{temp}'
        opFile = open(opFileName + '.txt', 'w')
        opFile.writelines('SMILES\tName\n')
        opFile.writelines(f'{seedSmi}\tSeed\n')   # file head
        for inputDic in inputList:
            seedSmi = inputDic['smi']
            molID = inputDic['molID']
            # IFP=np.array(IFP)
            IFP = np.array([IFP]*512)
            print(IFP)
            print(f'Sampling for molecule: {molID}')
            model.batch_input_length = 512
            # smiles, _ = model.predict_batch(latent=IFP, temp=0.5)
            # print(smiles)
            try:
                smiles, _ = model.predict_batch(latent=IFP, temp=temp)
                smiles = list(smiles)
                validity, valSmis = cal_valid(smiles)
                print(
                    f"index: {inputList.index(inputDic)}  validity: {validity}")

                for idx, smi in enumerate(valSmis):
                    opFile.writelines(
                        f'{smi}\t{molID}_sampled{idx}\n')
                opFile.close()
                mayaPath = '/mnt/home/zhangjie/Bin/mayachemtools/bin'
                os.system(
                    f'python {mayaPath}/RDKitDrawMoleculesAndDataTable.py -i {opFileName}.txt -s yes -o {opFileName}.html --infileParams "smilesDelimiter,tab,smilesColumn,1,smilesNameColumn,2"')
            except Exception as e:
                print(e)
                continue


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--IFP", help="IFP file to obtained seed randomly",
                        default='')
    parser.add_argument("--seed", help='csv file of seeds',
                        default='')
    parser.add_argument("--model", help="trained model",
                        default='')
    parser.add_argument("--save", help="file name for saving sampled SMILES (pkl)",
                        default='')
    parser.add_argument("--label", help="label of the result file",
                        type=str, default='sample')
    parser.add_argument("--switch_pt", help="the percentage of bits will be switched.",
                        type=float, default='0')
    args = parser.parse_args()
    return args


def main(args):
    temp = [0.2, 0.5, 1.0]
    savePath = Path(args.save)
    savePath.parent.mkdir(parents=True, exist_ok=True)
    benchmark_model(args, temp)  # generate SMILES from general model

    # generate SMILES from general model with drifted ecfp
    # temp = [1.0]
    # benchmark_efcpDrift(args, temp)

    # generate_smis(args)

    # sample_model(args, [1.0])


if __name__ == "__main__":
    args = get_parser()
    main(args)
