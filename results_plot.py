from __future__ import print_function
# import ddc.ddc_pub.ddc_v3 as ddc
import pandas as pd
import argparse
import numpy as np
import pickle
import seaborn as sns
import sys
import matplotlib.pyplot as plt
import os
from pathlib import Path
# from metrics_utils import logP, QED, SA, weight, NP
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED
import rdkit
from rdkit import Chem
import prettytable as ptb
# from ddc.ddc_pub import ddc_v3 as ddc


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    # os.system('mkdir obj')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def strip_pose(idx):
    idx_nopose = ''
    idx_sp = idx.split("_")
    if '_pose_' in idx:
        remove_num = 2
    else:
        remove_num = 1
    for itm in range(len(idx_sp)-remove_num):
        if idx_nopose == '':
            idx_nopose = idx_sp[itm]
        else:
            idx_nopose += f'_{idx_sp[itm]}'
    return idx_nopose


def df_shortname(df, col_name='id'):
    index_org = df.id
    short_name = [strip_pose(idx) for idx in index_org]
    # df['short_name'] = short_name
    return short_name


def read_log(log):
    res = []
    epoch = 0
    with open(log, "r", encoding='UTF-8') as f:
        line = 'line'
        while line:
            line = f.readline()
            # print(line)
            lineSp = line.split()
            # print(lineSp)
            if len(lineSp) == 12 and 'loss:' in lineSp and 'val_loss:' in lineSp:
                try:
                    val_loss = float(lineSp[8])
                    loss = float(lineSp[5])
                    epoch += 1
                    res.append([epoch, loss, val_loss])
                except Exception as e:
                    print(e)
    df = pd.DataFrame(res, columns=['Epoch', 'Loss', 'Val_loss'])
    return df


def best_value_pose(df, id='col name', value='col name'):
    '''A function to select best values from different poses'''
    # print(f'dfRes.keys={df.columns}  dfRes={df}')
    df = pd.DataFrame(df[[id, value]])
    short_name = df_shortname(df, id)
    df['short_name'] = short_name
    df[value] = df[value].astype(float)
    if value == 'dockScore':
        max_idx = df.groupby('short_name')[value].idxmin()
    else:
        max_idx = df.groupby('short_name')[value].idxmax()
    df = df.loc[max_idx]
    return df


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


def plot_IfpRecovery(sampledSmi, tempList):
    print('#'*10+"\nPloting the recovery of the IFP!")
    workPath = Path(sampledSmi)
    IfpRecList = []
    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')
        resList = load_obj(f"{sampledSmi}_{temp}_ifp")
        for resDic in resList:
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            dfRes = best_value_pose(dfRes, 'id', 'bitRec')
            print(f'dfRes.keys={dfRes.columns}  dfRes={dfRes}')
            for simItm in list(dfRes['bitRec']):
                IfpRecList.append([simItm, temp])
    resListReinvent = load_obj(f"{sampledSmi}_{1.0}_reinventIfpSim")
    resListActive = load_obj(f"{sampledSmi}_{1.0}_activeIfpSim")
    # resListTest = load_obj(f"{sampledSmi}_{tempList[-1]}_testIfpSim")
    resListRandomChembl = load_obj(
        f"{sampledSmi}_{1.0}_randomChemblIfpSim")
    for resDic in resListReinvent:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'bitRec')
        for simItm in list(dfRes['bitRec']):
            IfpRecList.append([simItm, 'REINVENT'])
    sampledSmi_tmp = sampledSmi.replace('1', '5')
    resListActive = load_obj(f"{sampledSmi_tmp}_{1.0}_activeIfpSim")
    for resDic in resListActive:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'bitRec')
        for simItm in list(dfRes['bitRec']):
            IfpRecList.append([simItm, 'Active'])
    # for resDic in resListTest:
    #     dfRes = resDic['dfRes']
    #     dfRes['id'] = dfRes.index
    #     dfRes = best_value_pose(dfRes, 'id', 'bitRec')
    #     for simItm in list(dfRes['bitRec']):
    #         IfpRecList.append([simItm, 'Test'])
    for resDic in resListRandomChembl:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'bitRec')
        for simItm in list(dfRes['bitRec']):
            IfpRecList.append([simItm, 'Random'])

    bitRecDf = pd.DataFrame(IfpRecList, columns=[
        'bitRecv', 'Temperature'])

    sns.set(style='ticks')
    plt.figure(figsize=(7, 5.4))
    for i in ['REINVENT', 'Active', 'Random'] + tempList:
        dfPlot = bitRecDf[bitRecDf['Temperature'] == i]
        plotData = list(dfPlot['bitRecv'])
        sns.distplot(plotData, kde=True, kde_kws={
                     "lw": 2, "label": i}, hist=False)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('IFP recovery', fontsize=14)
    plt.xlim(0,100)
    plt.ylabel('Density', fontsize=14)
    title = f'{sampledSmi}__IFP recovery'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.legend()
    imagePath = workPath.parent.joinpath('images')
    # os.system(f'mkdir {imagePath}')
    plt.savefig(
        os.path.join(imagePath, title+'.pdf')
    )
    plt.savefig(
        os.path.join(imagePath, title+'.png'),
        dpi=300
    )


def plot_IfpSimilarity(sampledSmi, tempList):
    print('#'*10+"\nPloting the IFP similarity!")
    workPath = Path(sampledSmi)
    IfpSimList = []
    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')
        resList = load_obj(f"{sampledSmi}_{temp}_ifp")
        for resDic in resList:
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            dfRes = best_value_pose(dfRes, 'id', 'ifpSim')
            print(f'dfRes.keys={dfRes.columns}  dfRes={dfRes}')
            for simItm in list(dfRes['ifpSim']):
                IfpSimList.append([simItm, temp])
    resListReinvent = load_obj(f"{sampledSmi}_{1.0}_reinventIfpSim")
    resListActive = load_obj(f"{sampledSmi}_{1.0}_activeIfpSim")
    resListRandomChembl = load_obj(
        f"{sampledSmi}_{1.0}_randomChemblIfpSim")
    for resDic in resListReinvent:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'ifpSim')
        for simItm in list(dfRes['ifpSim']):
            IfpSimList.append([simItm, 'REINVENT'])
    sampledSmi_tmp = sampledSmi.replace('1', '5')
    resListActive = load_obj(f"{sampledSmi_tmp}_{1.0}_activeIfpSim")
    for resDic in resListActive:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'ifpSim')
        for simItm in list(dfRes['ifpSim']):
            IfpSimList.append([simItm, 'Active'])
    for resDic in resListRandomChembl:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'ifpSim')
        for simItm in list(dfRes['ifpSim']):
            IfpSimList.append([simItm, 'Random'])

    IfpSimDf = pd.DataFrame(IfpSimList, columns=[
        'ifpSim', 'Temperature'])

    sns.set(style='ticks')
    plt.figure(figsize=(7, 5.4))
    for i in ['REINVENT', 'Active', 'Random'] + tempList:
        dfPlot = IfpSimDf[IfpSimDf['Temperature'] == i]
        IfpSimData = list(dfPlot['ifpSim'])
        sns.distplot(IfpSimData, kde=True, kde_kws={
                     "lw": 2, "label": i}, hist=False)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('IFP tanimoto similarity', fontsize=14)
    plt.xlim(0,1)
    plt.ylabel('Density', fontsize=14)
    title = f'{sampledSmi}__IFP tanimoto similarity'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.legend()
    imagePath = workPath.parent.joinpath('images')
    os.system(f'mkdir {imagePath}')
    plt.savefig(
        os.path.join(imagePath, title+'.png'),
        dpi=300
    )
    plt.savefig(
        os.path.join(imagePath, title+'.pdf')
    )


def plot_dockScore(sampledSmi, tempList):
    print('#'*10+"\nPloting the docking score!")
    workPath = Path(sampledSmi)
    dockScoreList = []
    resListReinvent = load_obj(f"{sampledSmi}_{1.0}_reinventIfpSim")
    # resListActive = load_obj(f"{sampledSmi}_{tempList[0]}_activeIfpSim")
    resListActive = load_obj(f"{sampledSmi}_{1.0}_activeIfpSim")
    resListRandomChembl = load_obj(
        f"{sampledSmi}_{1.0}_randomChemblIfpSim")
    for resDic in resListReinvent:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'dockScore')
        for simItm in list(dfRes['dockScore']):
            dockScoreList.append([simItm, 'REINVENT'])
    for resDic in resListActive:
        dfRes = resDic['dfRes']
        print(dfRes)
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'dockScore')
        for simItm in list(dfRes['dockScore']):
            dockScoreList.append([simItm, 'Active'])
    for resDic in resListRandomChembl:
        dfRes = resDic['dfRes']
        dfRes['id'] = dfRes.index
        dfRes = best_value_pose(dfRes, 'id', 'dockScore') # 'score_0' sometimes
        for simItm in list(dfRes['dockScore']):
            dockScoreList.append([simItm, 'Random'])
    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')
        resList = load_obj(f"{sampledSmi}_{temp}_ifp")
        for resDic in resList:
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            dfRes = best_value_pose(dfRes, 'id', 'dockScore')
            for simItm in list(dfRes['dockScore']):
                dockScoreList.append([simItm, temp])

    IfpSimDf = pd.DataFrame(dockScoreList, columns=[
        'dockScore', 'Temperature'])

    sns.set(style='ticks')
    plt.figure(figsize=(7, 5.4))
    for i in ['REINVENT', 'Active', 'Random'] + tempList:
        dfPlot = IfpSimDf[IfpSimDf['Temperature'] == i]
        IfpSimData = list(dfPlot['dockScore'])
        sns.distplot(IfpSimData, kde=True, kde_kws={
                     "lw": 2, "label": i}, hist=False)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Docking score', fontsize=14)
    plt.xlim(-14, -2)
    plt.ylabel('Density', fontsize=14)
    title = f'{sampledSmi}__Docking score'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.legend()
    imagePath = workPath.parent.joinpath('images')
    os.system(f'mkdir {imagePath}')
    plt.savefig(
        os.path.join(imagePath, title+'.pdf')
    )
    plt.savefig(
        os.path.join(imagePath, title+'.png'),
        dpi=300
    )


def plot_molSimilarity(sampledSmi, tempList):
    print('#'*10+"\nPloting the molecular similarity!")
    workPath = Path(sampledSmi)
    molSimList = []
    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')
        resList = load_obj(f"{sampledSmi}_{temp}_ifp")
        for resDic in resList:
            dfRes = resDic['dfRes']
            for simItm in list(dfRes['molSim']):
                molSimList.append([simItm, temp])
    resListReinvent = load_obj(f"{sampledSmi}_{1.0}_reinventIfpSim")
    resListActive = load_obj(f"{sampledSmi}_{1.0}_activeIfpSim")
    resListRandomChembl = load_obj(
        f"{sampledSmi}_{1.0}_randomChemblIfpSim")
    for resDic in resListReinvent:
        dfRes = resDic['dfRes']
        for simItm in list(dfRes['molSim']):
            molSimList.append([simItm, 'REINVENT'])
    sampledSmi_tmp = sampledSmi.replace('1', '5')
    resListActive = load_obj(f"{sampledSmi_tmp}_{1.0}_activeIfpSim")
    for resDic in resListActive:
        dfRes = resDic['dfRes']
        for simItm in list(dfRes['molSim']):
            molSimList.append([simItm, 'Active'])
    for resDic in resListRandomChembl:
        dfRes = resDic['dfRes']
        for simItm in list(dfRes['molSim']):
            molSimList.append([simItm, 'Random'])
    molSimDf = pd.DataFrame(molSimList, columns=[
        'molSim', 'Temperature'])

    sns.set(style='ticks')
    plt.figure(figsize=(7, 5.4))
    for i in ['REINVENT', 'Active', 'Random'] + tempList:
        dfPlot = molSimDf[molSimDf['Temperature'] == i]
        molSimData = list(dfPlot['molSim'])
        sns.distplot(molSimData, kde=True, kde_kws={
                     "lw": 2, "label": i}, hist=False)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Molecular tanimoto similarity', fontsize=14)
    plt.xlim(0,1)

    plt.ylabel('Density', fontsize=14)
    title = f'{sampledSmi}__Molecular tanimoto similarity'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.legend()
    imagePath = workPath.parent.joinpath('images')
    os.system(f'mkdir {imagePath}')
    plt.savefig(
        os.path.join(imagePath, title+'.pdf')
    )
    plt.savefig(
        os.path.join(imagePath, title+'.png'),
        dpi=300
    )


def plot_validity(sampledSmi, tempList):
    print('#'*10+"\nPloting the validity!")
    workPath = Path(sampledSmi)
    validityList = []
    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}.pkl!')
        resList = load_obj(f"{sampledSmi}_{temp}")
        for resDic in resList:
            try:
                resDic['validity'] = float(resDic['validity'])
                validityList.append([resDic['validity'], temp])
            except:
                print(e)
                continue
    validityDf = pd.DataFrame(validityList, columns=[
                              'validity', 'Temperature'])
    sns.set(style='ticks')
    plt.figure(figsize=(7, 5.4))
    # sns.displot(data=validityDf, x='validity',
    #             hue='Temperature', kind='kde', fill=True, linewidth=2)
    for i in tempList:
        dfPlot = validityDf[validityDf['Temperature'] == i]
        molSimData = list(dfPlot['validity'])
        sns.distplot(molSimData, kde=True, kde_kws={
                     "lw": 2, "label": i}, hist=False)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Validity of generated SMILES (%)', fontsize=14)
    plt.xlim(30, 100)
    # plt.ylim(0, 1.5)
    plt.ylabel('Density', fontsize=14)
    title = f'{sampledSmi}__Validity_{workPath.stem}'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.legend()
    imagePath = workPath.parent.joinpath('images')
    os.system(f'mkdir {imagePath}')
    plt.savefig(
        os.path.join(imagePath, title+'.pdf')
    )
    plt.savefig(
        os.path.join(imagePath, title+'.png'),
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


def plot_dscore_drift(sampledSmi, tempList):
    print('#'*10+"\nPloting the drift of the properties!")
    workPath = Path(sampledSmi)
    PPList = []
    temp = 1.0
    iseed = 1
    resListReinvent = load_obj(f"{sampledSmi}_{temp}_reinventIfpSim")
    resListActive = load_obj(f"{sampledSmi}_{temp}_activeIfpSim")
    resListRandomChembl = load_obj(
        f"{sampledSmi}_{temp}_randomChemblIfpSim")
    dfRes = resListReinvent[0]['dfRes']
    dfRes['id'] = dfRes.index
    dfRes = best_value_pose(dfRes, 'id', 'dockScore')
    for dockScore in list(dfRes['dockScore']):
        PPList.append([dockScore, 'REINVENT'])
    dfRes = resListActive[0]['dfRes']
    dfRes['id'] = dfRes.index
    dfRes = best_value_pose(dfRes, 'id', 'dockScore')
    for dockScore in list(dfRes['dockScore']):
        PPList.append([dockScore, 'Active'])
    dfRes = resListRandomChembl[0]['dfRes']
    dfRes['id'] = dfRes.index
    dfRes = best_value_pose(dfRes, 'id', 'dockScore')
    for dockScore in list(dfRes['dockScore']):
        PPList.append([dockScore, 'Random'])

    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')

        resList = load_obj(f"{sampledSmi}_{temp}_ifp")
        # print(resList)
        for resDic in [resList[iseed]]:
            # print(resDic)
            seedSmi = resDic['seedSmi']
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            dfRes = best_value_pose(dfRes, 'id', 'dockScore')
            for dockScore in list(dfRes['dockScore']):
                PPList.append([dockScore, temp])

    IfpSimDf = pd.DataFrame(PPList, columns=[
        'dockScore', 'Temperature'])

    sns.set(style='ticks')
    plt.figure(figsize=(7, 5.4))
    for i in ['REINVENT', 'Active', 'Random'] + tempList:
        dfPlot = IfpSimDf[IfpSimDf['Temperature'] == i]
        IfpSimData = list(dfPlot['dockScore'])
        sns.distplot(IfpSimData, kde=True, kde_kws={
                     "lw": 2, "label": i}, hist=False)

    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    #  markers=True, style='Type')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel(f'Docking score (seed: -10.4)', fontsize=14)
    plt.xlim(-12, -4.5)
    plt.ylim(0, 1.8)
    plt.ylabel('Density', fontsize=14)
    title = f'{sampledSmi}_1.0_dScore'
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.legend()
    imagePath = workPath.parent.joinpath('images')
    os.system(f'mkdir {imagePath}')
    plt.savefig(
        os.path.join(imagePath, title+'.pdf')
    )
    plt.savefig(
        os.path.join(imagePath, title+'.png'),
        dpi=300
    )


def get_pProperty(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        molwt = Descriptors.ExactMolWt(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        qed = QED.qed(mol)
        # logp = logP(mol)
        # qed = QED(mol)
        # sa = SA(mol)
        # wt = weight(mol)
        # np = NP(mol)
        return [logp, tpsa, molwt, hba, hbd, qed]
    except Exception as e:
        print(e)
        return None


def plot_pproperty_drift(sampledSmi, tempList):
    print('#'*10+"\nPloting the drift of the properties!")
    workPath = Path(sampledSmi)
    PPList = []
    temp = 1.0
    iseed = 1
    resListReinvent = load_obj(f"{sampledSmi}_{temp}_reinventIfpSim")
    resListActive = load_obj(f"{sampledSmi}_{temp}_activeIfpSim")
    resListRandomChembl = load_obj(
        f"{sampledSmi}_{temp}_randomChemblIfpSim")
    for smi in list(resListReinvent[0]['dfRes']['smi']):
        pp = get_pProperty(smi)
        if pp:
            PPList.append(pp+['REINVENT'])
    for smi in list(resListActive[0]['dfRes']['smi']):
        pp = get_pProperty(smi)
        if pp:
            PPList.append(pp+['Active'])
    for smi in list(resListRandomChembl[0]['dfRes']['smi']):
        pp = get_pProperty(smi)
        if pp:
            PPList.append(pp+['Random'])

    for temp in tempList:
        print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')

        resList = load_obj(f"{sampledSmi}_{temp}_ifp")
        # print(resList)
        for resDic in [resList[iseed]]:
            # print(resDic)
            seedSmi = resDic['seedSmi']
            dfRes = resDic['dfRes']
            # print(dfRes)
            seedPP = get_pProperty(seedSmi)
            print(f"seed = {seedSmi}, {seedPP}")
            for smi in list(resDic['smis']):
                pp = get_pProperty(smi)
                if pp:
                    PPList.append(pp+[temp])

    IfpSimDf = pd.DataFrame(PPList, columns=[
        'logP', 'TPSA', 'MW', 'HBA', 'HBD', 'QED', 'Temperature'])

    sns.set(style='ticks')
    for ppItm in ['logP', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']:
        plt.figure(figsize=(7, 5.4))
        for i in ['REINVENT', 'Active', 'Random'] + tempList:
            dfPlot = IfpSimDf[IfpSimDf['Temperature'] == i]
            IfpSimData = list(dfPlot[ppItm])
            sns.distplot(IfpSimData, kde=True, kde_kws={
                         "lw": 2, "label": i}, hist=False)

        plt.rc('font', family='Times New Roman', size=12, weight='bold')
        paper_rc = {'lines.linewidth': 8, 'lines.markersize': 8}
        sns.set_context("paper", rc=paper_rc)
        #  markers=True, style='Type')
        # g.set(xticklabels=[])
        # plt.yscale('log')
        ppName = ['logP', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']
        seedValue = round(seedPP[ppName.index(ppItm)], 2)
        plt.xlabel(f'{ppItm} (seed: {seedValue})', fontsize=14)
        if ppItm == 'HBA':
            plt.xlim(-1, 15)
            # plt.ylim(0, 3)
        if ppItm == 'HBD':
            plt.xlim(-1, 10)
            # plt.ylim(0, 40)
        if ppItm == 'logP':
            plt.xlim(-3, 10)
            # plt.ylim(0, 8)
        if ppItm == 'MW':
            plt.xlim(0, 800)
            # plt.ylim(0, 0.15)
        if ppItm == 'TPSA':
            plt.xlim(-20, 250)
        #     plt.ylim(0, 6)
        if ppItm == 'QED':
            plt.xlim(-0.1, 1.1)
        #     plt.ylim(0, 6)
        plt.ylabel('Density', fontsize=14)
        title = f'{sampledSmi}_1.0_{ppItm}'
        # plt.title(title, fontsize=14)
        plt.tight_layout()
        plt.legend()
        imagePath = workPath.parent.joinpath('images')
        os.system(f'mkdir {imagePath}')
        plt.savefig(
            os.path.join(imagePath, title+'.pdf')
        )
        plt.savefig(
            os.path.join(imagePath, title+'.png'),
            dpi=300
        )


def cal_average_performance(sampledSmi, tempList, load_pkl=False):
    '''Average performance will be evaluated based on the data in the ifp files.'''
    if load_pkl:
        molSimDf = load_obj(f'{sampledSmi}_resultDf')
    else:
        print('#'*10+"\n Loading the results!")
        workPath = Path(sampledSmi)
        molSimList = []
        value_names = ['bitRec', 'ifpSim', 'dockScore']
        for temp in tempList:
            resList = load_obj(f"{sampledSmi}_{temp}_ifp")
            print(f'Loading {sampledSmi}_{temp}_ifp.pkl!')
            for resDic in resList:
                molSimList.append(
                    [float(resDic['validity']), temp, 'validity'])
                dfRes = resDic['dfRes']
                for simItm in list(dfRes['molSim']):
                    molSimList.append([simItm, temp, 'molSim'])
                dfRes['id'] = dfRes.index
                for ivalue in value_names:
                    dfRes_best = dfRes.copy(deep=True)
                    dfRes_best = best_value_pose(dfRes_best, 'id', ivalue)
                    for simItm in list(dfRes_best[ivalue]):
                        molSimList.append([simItm, temp, ivalue])
        resListReinvent = load_obj(
            f"{sampledSmi}_{tempList[-1]}_reinventIfpSim")
        resListActive = load_obj(f"{sampledSmi}_{tempList[-1]}_activeIfpSim")
        resListRandomChembl = load_obj(
            f"{sampledSmi}_{tempList[-1]}_randomChemblIfpSim")

        for resDic in resListReinvent:
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            for simItm in list(dfRes['molSim']):
                molSimList.append([simItm, 'REINVENT', 'molSim'])
            for ivalue in value_names:
                dfRes_best = dfRes.copy(deep=True)
                dfRes_best = best_value_pose(dfRes_best, 'id', ivalue)
                for simItm in list(dfRes_best[ivalue]):
                    molSimList.append([simItm, 'REINVENT', ivalue])

        for resDic in resListActive:
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            for simItm in list(dfRes['molSim']):
                molSimList.append([simItm, 'Active', 'molSim'])
            for ivalue in value_names:
                dfRes_best = dfRes.copy(deep=True)
                dfRes_best = best_value_pose(dfRes_best, 'id', ivalue)
                for simItm in list(dfRes_best[ivalue]):
                    molSimList.append([simItm, 'Active', ivalue])

        for resDic in resListRandomChembl:
            dfRes = resDic['dfRes']
            dfRes['id'] = dfRes.index
            for simItm in list(dfRes['molSim']):
                molSimList.append([simItm, 'Random', 'molSim'])
            for ivalue in value_names:
                dfRes_best = dfRes.copy(deep=True)
                dfRes_best = best_value_pose(dfRes_best, 'id', ivalue)
                for simItm in list(dfRes_best[ivalue]):
                    molSimList.append([simItm, 'Random', ivalue])

        molSimDf = pd.DataFrame(molSimList, columns=['value', 'Temperature',
                                                     'item'])
        save_obj(molSimDf, f'{sampledSmi}_resultDf')
    molSimDf['Temperature'] = molSimDf['Temperature'].astype(str)
    molSimDf['value'] = molSimDf['value'].astype(float)
    molSimDf = molSimDf.dropna(how='any', axis=0)
    molSimDf = molSimDf.set_index(['item', 'Temperature'])
    results_sum = []

    for itm in ['bitRec', 'dockScore',  'ifpSim',  'molSim']:
        for temp in ['0.2', '0.5', '1.0', 'Active', 'REINVENT', 'Random']:
            df_tmp = molSimDf.loc[(itm, temp)]
            value_list = df_tmp['value']
            value_np = np.array(list(value_list))
            value_np = np.sort(value_np)
            print(f'value_np= {value_np}')
            top10_idx = int(len(value_list)*0.05)
            if top10_idx < 1:
                top10_idx = 1
            if itm == 'dockScore':
                top_mean = value_np[:top10_idx].mean()
            else:
                top_mean = value_np[-top10_idx:].mean()
            results_sum.append([itm, temp, value_np.mean(), top_mean])
    for temp in ['0.2', '0.5', '1.0']:
        df_tmp = molSimDf.loc[('validity', temp)]
        value_list = df_tmp['value']
        value_np = np.array(list(value_list))
        results_sum.append(
            ['validity', temp, value_np.mean(), value_np.mean()])

    result_df = pd.DataFrame(results_sum, columns=[
        'Item', 'Kind', 'Average', 'Top5% Average'])
    # result_df['Top5% Average'] = result_df['Top5% Average'].round(decimals=2)
    # result_df['Average'] = result_df['Average'].round(decimals=2)
    # result_df = result_df['Top5% Average'].astype(float)
    result_df = result_df.round(decimals=2)
    # pd.set_option('precision', 2)
    print(result_df)
    result_df.to_csv(f'{sampledSmi}_result_average.csv', index=False)


def compare_models():
    '''Compare models and please run in the working directory!'''
    models_eval = ['AIFPsmi_5pose', 'dScore_1pose',
                   'dScore_5pose', 'ECFPsmi_5pose', 'resAIFPsmi_5pose']
    result_dfs = []
    for imodel in models_eval:
        idf = pd.read_csv(f'{imodel}_result_average.csv')
        idf['Kind'] = idf['Kind'].astype(str)
        idf = idf.set_index(['Item', 'Kind'])
        result_dfs.append(idf)
    result_sum = []
    for itm in ['bitRec', 'dockScore',  'ifpSim',  'molSim']:
        for temp in ['0.2', '0.5', '1.0', 'Active', 'REINVENT', 'Random']:
            result_model = [itm, temp]
            for idf in result_dfs:
                result_model.append(idf.loc[(itm, temp), 'Average'])
            for idf in result_dfs:
                result_model.append(idf.loc[(itm, temp), 'Top5% Average'])
            result_sum.append(result_model)
    result_df = pd.DataFrame(result_sum, columns=[
        'Item', 'Kind', 'AIFPsmi_5pose', 'dScore_1pose',
        'dScore_5pose', 'ECFPsmi_5pose', 'resAIFPsmi_5pose', 'AIFPsmi_5pose', 'dScore_1pose',
        'dScore_5pose', 'ECFPsmi_5pose', 'resAIFPsmi_5pose'])
    print(result_df)
    result_df.to_csv('models_compare_summary.csv', index=False)


def main(args):
    sampledSmi = args.sampledSmi
    # tempList = [0.2, 0.5, 1.0]
    tempList = [0.5]
    # plot_validity(sampledSmi, tempList)
    plot_molSimilarity(sampledSmi, tempList)
    plot_IfpSimilarity(sampledSmi, tempList)
    plot_IfpRecovery(sampledSmi, tempList)
    plot_dockScore(sampledSmi, tempList)

    # plot_dscore_drift(sampledSmi, tempList)
    # plot_pproperty_drift(sampledSmi, tempList)

    # cal_average_performance(sampledSmi, tempList, load_pkl=False)
    # compare_models()


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--log", help="log file",
    #                     default='')
    # parser.add_argument("--img_folder", help="image folder",
    #                     default='./images')
    parser.add_argument("--sampledSmi", help="sampled smiles",
                        default='')
    # parser.add_argument("--sampledIFP", help="IFP of sampled smiles",
    #                     default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
