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
# import rdkit
# from rdkit import Chem
# from ddc.ddc_pub import ddc_v3 as ddc


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", help="log file",
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


def main(args):
    dfLog = read_log(args.log)
    print(dfLog)
    dfPlot = to_plotDf(dfLog)
    dfPlot.to_csv(args.log.replace('.txt', '_trainLoss.csv'), index=None)
    print(dfPlot)
    line_plot(dfPlot, args)


if __name__ == "__main__":
    args = get_parser()
    main(args)
