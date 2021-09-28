from __future__ import print_function
# import ddc.ddc_pub.ddc_v3 as ddc
import pandas as pd
import argparse
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from metrics_utils import logP, QED, SA, weight
# from ddc.ddc_pub import ddc_v3 as ddc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampledSmi", help="pkl file of sampled smiles",
                        default='')
    parser.add_argument("--img_folder", help="image folder",
                        default='./images')
    args = parser.parse_args()
    return args


def save_obj(obj, name):
    os.system('mkdir obj')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def read_csv(csv):
    res = []

    with open(csv, "r") as f:
        line = 'line'
        while line:
            line = f.readline()
            # print(line)
            lineSp = line.split()
            try:
                if lineSp[0] == 'smi:':
                    orgSmi = lineSp[1]
                    orgId = lineSp[3]
                    orgBits = lineSp[5:]
                else:
                    res.append(lineSp[0])
            except Exception as e:
                print(e)

    df = pd.DataFrame(res, columns=['Smi'])
    return {'df': df, 'orgSmi': orgSmi, "orgId": orgId, "orgBits": orgBits}


def to_plotDf(df):
    newList = []   # [Item,Value,type]
    for idx, col in df.iterrows():
        Epoch = col['Epoch']
        Loss = col['Loss']
        Val_loss = col['Val_loss']
        newList.append([Epoch, Loss, 'Training loss'])
        newList.append([Epoch, Val_loss, 'Validation loss'])
    newDf = pd.DataFrame(newList, columns=['Epoch', 'Loss_value', 'Type'])
    return newDf


def line_plot(df, args):
    os.system('mkdir images')
    sns.set(style='ticks')
    plt.figure(figsize=(7, 4.8))
    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    ifg = df
    paper_rc = {'lines.linewidth': 2, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    sns.lineplot(x='Epoch', y='Loss_value', data=ifg, hue='Type')
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    plt.xlim(0, 600)
    plt.ylabel('Loss', fontsize=14)
    logPath = Path(args.log)
    logName = logPath.name
    title = logName.replace('.txt', '_trainLoss')
    plt.savefig(
        os.path.join(args.img_folder, title+'.pdf')
    )
    plt.savefig(
        os.path.join(args.img_folder, title+'.png'),
        dpi=250
    )


def cal_validity(sampledSmi, tempList):
    validityList = []
    for temp in tempList:
        resList = load_obj(f"{sampledSmi}{temp}")
        for resDic in resList:
            validityList.append([resDic['validity'], temp])
    validityDf = pd.DataFrame(validityList, columns=[
                              'validity', 'Temperature'])
    sns.set(style='ticks')
    # plt.figure(figsize=(7, 5.4))
    sns.displot(data=validityDf, x='validity',
                hue='Temperature', kind='kde', fill=True, linewidth=2)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Validity of generated SMILES', fontsize=14)
    plt.xlim(75, 110)
    plt.ylabel('Density', fontsize=14)
    title = f'Validity of generated SMILES Epoch 400'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    # plt.legend('middle left')
    plt.savefig(
        os.path.join('images', title+'.pdf')
    )
    plt.savefig(
        os.path.join('images', title+'.png'),
        dpi=300
    )


def smis_toMol(smiList):
    molList = []
    FPList = []
    for smi in smiList:
        try:
            mol = Chem.MolFromSmiles(smi)
            molList.append(mol)
            FPList.append(AllChem.GetMorganFingerprintAsBitVect(
                mol, 3, nBits=1024))
            # FPList.append(Chem.RDKFingerprint(mol))
        except Exception as e:
            print(e)
            continue
    return molList, FPList


def cal_molSimilarity(sampledSmi, tempList):
    ifCalFP = True
    if ifCalFP:
        molSimList = []
        for temp in tempList:
            resList = ''
            resList = load_obj(f"{sampledSmi}{temp}")
            for resDic in resList:
                smiList = resDic['smis']
                seedSmi = resDic['seedSmi']
                seedMol = Chem.MolFromSmiles(seedSmi)
                seedFP = AllChem.GetMorganFingerprintAsBitVect(
                    seedMol, 3, nBits=1024)
                # seedFP=Chem.RDKFingerprint(seedMol)
                Mols, FPList = smis_toMol(smiList)
                print(f"temp: {temp}, idx: {resList.index(resDic)}")
                molSims = [DataStructs.TanimotoSimilarity(
                    FPItm, seedFP) for FPItm in FPList]
                # print(molSims)s
                print(len(molSims))
                print(molSims)
                for simItm in molSims:
                    molSimList.append([simItm, temp])
        molSimDf = pd.DataFrame(molSimList, columns=[
            'molSim', 'Temperature'])
        save_obj(molSimDf, 'molSimDf')
    else:
        molSimDf = load_obj('molSimDf')
        print(molSimDf)
    sns.set(style='ticks')
    # plt.figure(figsize=(7, 5.4))
    sns.displot(data=molSimDf, x='molSim',
                hue='Temperature', kind='kde', fill=True, linewidth=2)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Molecular tanimoto similarity', fontsize=14)
    plt.xlim(-0.05, 0.45)
    plt.ylabel('Density', fontsize=14)
    title = f'Molecular tanimoto similarity 400'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    # plt.legend('middle left')
    plt.savefig(
        os.path.join('images', title+'.pdf')
    )
    plt.savefig(
        os.path.join('images', title+'.png'),
        dpi=300
    )


def feats_cal(smiDf):
    res = []
    for idx, row in smiDf.iterrows():
        try:
            smi = row['Smi']
            mol = Chem.MolFromSmiles(smi)
            logp = logP(mol)
            qed = QED(mol)
            sa = SA(mol)
            wt = weight(mol)
        except Exception as e:
            print(e)
            continue
        res.append([smi, logp, qed, sa, wt])
    df = pd.DataFrame(res, columns=['Smi', 'logP', 'QED', 'SA', 'Weight'])
    return df


def dist_plot(df, dicRes):
    seedSmi = dicRes['orgSmi']
    mol = Chem.MolFromSmiles(seedSmi)
    logp = logP(mol)
    qed = QED(mol)
    sa = SA(mol)
    wt = weight(mol)

    seedFeats = {'logP': logp, 'QED': qed, 'SA': sa, 'Weight': wt}
    Itms = ['logP', 'QED', 'SA', 'Weight']
    for itm in Itms:
        ifg = df[itm]
        os.system('mkdir images')
        sns.set(style='ticks')
        plt.figure(figsize=(7, 4.8))
        plt.rc('font', family='Times New Roman', size=12, weight='bold')
        paper_rc = {'lines.linewidth': 2, 'lines.markersize': 8}
        sns.set_context("paper", rc=paper_rc)
        sns.distplot(ifg, hist=True, kde=True,
                     kde_kws={'shade': True, 'linewidth': 2}, label='Seed %s: %.2f' % (itm, seedFeats[itm]))
        #  markers=True, style='Type')
        # g.set(xticklabels=[])
        # plt.yscale('log')
        plt.xlabel(itm, fontsize=14)
        # plt.xlim(0, 600)
        plt.ylabel('Density', fontsize=14)
        id = dicRes['orgId']
        title = f'{id}_{itm}'
        plt.title(title, fontsize=14)
        plt.legend()
        plt.savefig(
            os.path.join(args.img_folder, title+'.pdf')
        )
        plt.savefig(
            os.path.join(args.img_folder, title+'.png'),
            dpi=300
        )


def main(args):
    ifCalValid = False
    CalMolSimilarity = True
    tempList = [0.2, 0.4, 0.6, 0.8, 1.0]
    sampleSmi = 'model_234actfullBits_400_sampledSmi_Temp'
    if ifCalValid:
        cal_validity(sampleSmi, tempList)
    if CalMolSimilarity:
        cal_molSimilarity(sampleSmi, [1.0]])

        # dfSmi = dicRes['df']
        # validRate = valid_cal(dfSmi)
        # dfFeats = feats_cal(dfSmi)
        # dist_plot(dfFeats, dicRes)
        # print(dfLog)

        # dfPlot = to_plotDf(dfLog)
        # dfPlot.to_csv(args.log.replace('.txt', '_trainLoss.csv'), index=None)
        # print(dfPlot)
        # line_plot(dfPlot, args)


if __name__ == "__main__":
    args = get_parser()
    main(args)
