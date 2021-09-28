import random
from scipy.stats import wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--IFP_file", help="IFP file",
                        default='')
    parser.add_argument("--AV_file", help="Activity file",
                        default='./Dataset/data_dock_score.csv')
    parser.add_argument("--img_folder", help="Images folder",
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


def analyze_IFP(df_ifp, args):
    '''
    Compress IFP to remove not important descriptors
    '''
    # df_ifp = df_ifp.set_index('Molecule')
    count_col = df_ifp.astype('int').sum()
    count_col.sort_values(ascending=False, inplace=True)
    pd_count_col = pd.DataFrame(count_col, columns=['counts'])
    pd_count_col.to_csv(args.IFP_file.replace('.csv', '_bits_count.csv'))
    os.system('mkdir images')
    sns.set(style='ticks')
    plt.figure(figsize=(7, 4.8))
    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    ifg = pd_count_col
    ifg['index'] = range(len(ifg))
    ifg['style'] = 'Bits counts'
    print(ifg)
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    sns.lineplot(x='index', y='counts', data=ifg,
                 markers=True, style='style')
    # g.set(xticklabels=[])
    plt.yscale('log')
    plt.xlabel('IFP bits', fontsize=14)
    plt.xlim(0, 125)
    plt.ylabel('Counts', fontsize=14)
    title = args.IFP_file.replace('.csv', '_bitscount')
    plt.savefig(
        os.path.join(args.img_folder, title+'.pdf')
    )
    plt.savefig(
        os.path.join(args.img_folder, title+'.png'),
        dpi=250
    )


def get_AV(df_ifp, df_AV):
    for index in df_ifp.index:
        try:
            df_ifp.loc[index, 'logValue'] = df_AV.loc[index, 'logValue']
            df_ifp.loc[index, 'score_0'] = df_AV.loc[index, 'score_0']
            # df_ifp.loc[index, 'Type'] = df_AV.loc[index, 'Type']
            # select not null of logValue
        except Exception as e:
            print(e)
    df_ifp.dropna(subset=['logValue', 'score_0'], inplace=True)
    df_ifp.sort_values(by='logValue', ascending=True, inplace=True)
    return df_ifp


def compress_IFP(df_ifp, num, args):
    '''
    Compress IFP to remove not important descriptors
    '''

    count_col = df_ifp.astype('int').sum()
    count_col.sort_values(ascending=False, inplace=True)
    pd_count_col = pd.DataFrame(count_col, columns=['counts'])
    pd_count_col.to_csv('Interaction_occurrence_count.csv')
    top_col = count_col[range(num)]
    top_AAIFP = df_ifp[top_col.index]
    return top_AAIFP


def create_nonzeroRefer(df_ifp, args, referName):
    '''
    create refer that the columns of not all zeros
    '''
    # count_col = count_col.set_index('Molecule')
    # count_col['Molecule'] = ''
    countDf = df_ifp.copy()
    countDf = countDf.drop(['smi'], axis=1)
    count_col = countDf.astype('int').sum()
    count_col.sort_values(ascending=False, inplace=True)
    pd_count_col = pd.DataFrame(count_col, columns=['counts'])
    pd_count_col = pd_count_col[pd_count_col['counts'] > 0]
    save_obj(list(pd_count_col.index), referName)


def remove_allzeros(df_ifp, args, referName):
    refer_nonzero = load_obj(referName)
    refer_nonzero += ['smi']
    nonzero_IFP = df_ifp[refer_nonzero]
    return nonzero_IFP


def heatmap_plot(df_og, args):
    dfSize = len(df_og)
    step = dfSize//5
    df = df_og.drop(['logValue', 'score_0'], axis=1)
    for i in range(5):
        # i += 1
        # index = range()
        ifg = df[i*step:i*step+30]
        ifg_og = df_og[i*step:i*step+30]
        plt.subplot(510+i+1)
        os.system('mkdir images')
        sns.set(style='ticks')
        # plt.figure(figsize=(7, 4.8))
        plt.rc('font', family='Times New Roman', size=12, weight='bold')
        paper_rc = {'lines.linewidth': 1, 'lines.markersize': 8}
        sns.set_context("paper", rc=paper_rc)
        sns.heatmap(ifg, cmap='viridis')
        plt.xticks([])
        lenIfg = len(ifg)
        plt.yticks([0, len(ifg)], ['%0.2f' % (ifg_og.iloc[0]['logValue']),
                                   '%0.2f' % (ifg_og.iloc[lenIfg-1]['logValue'])])
        if i == 2:
            plt.ylabel('Ligands with specific pIC50', fontsize=14)
        else:
            plt.ylabel('')
    tk_ndim = ifg.shape[1]
    xlabels = np.arange(0, tk_ndim, tk_ndim//10)
    plt.xticks(xlabels, xlabels)
    plt.xlabel('IFP bits', fontsize=14)

    title = args.IFP_file.replace('.csv', f'_heatmap_IFP')
    plt.savefig(
        os.path.join(args.img_folder, title+'.pdf')
    )
    plt.savefig(
        os.path.join(args.img_folder, title+'.png'),
        dpi=250
    )


def bitViolin_plot(df_og, args):
    os.system('mkdir images')
    dfSize = len(df_og)
    step = dfSize//6
    df = df_og.drop(['logValue', 'score_0'], axis=1)
    df['counts'] = df.apply(lambda x: x.sum(), axis=1)
    combineDf = ''
    for i in range(5):
        # i += 1
        # index = range()
        ifg = df[i*step:(i+1)*step]
        lenIfg = len(ifg)
        ifg_og = df_og[i*step:(i+1)*step]
        # counts
        ifg_count = ifg['counts']
        ifg_count = pd.DataFrame(ifg_count)
        ifg_count['type'] = 'Bit counts'
        ifg_count['set'] = "%0.2f_%0.2f" % (
            ifg_og.iloc[0]['logValue'], ifg_og.iloc[lenIfg-1]['logValue'])
        ifg_count = ifg_count[['counts', 'type', 'set']]
        ifg_count.columns = ['Value', 'Type', 'Set']
        # pIC50
        ifg_pic = ifg_og['logValue']
        ifg_count = pd.DataFrame(ifg_count)
        ifg_count['type'] = 'Bit counts'
        ifg_count['set'] = "%0.2f_%0.2f" % (
            ifg_og.iloc[0]['logValue'], ifg_og.iloc[lenIfg-1]['logValue'])
        ifg_count = ifg_count[['counts', 'type', 'set']]
        ifg_count.columns = ['Value', 'Type', 'Set']

        ifg[['logValue', 'score_0']] = ifg_og[['logValue', 'score_0']]
        newList = []
        # '%0.2f' % (ifg_og.iloc[0]['logValue']),
        # '%0.2f' % (ifg_og.iloc[lenIfg-1]['logValue'])]

        if combineDf == '':
            combineDf = ifg
        else:
            combineDf = pd.concat([combineDf, ifg])

        sns.set(style='ticks')
        # plt.figure(figsize=(7, 4.8))
        plt.rc('font', family='Times New Roman', size=12, weight='bold')
        paper_rc = {'lines.linewidth': 1, 'lines.markersize': 8}
        sns.set_context("paper", rc=paper_rc)
        sns.heatmap(ifg, cmap='viridis')
        plt.xticks([])
        lenIfg = len(ifg)
        plt.yticks([0, len(ifg)], ['%0.2f' % (ifg_og.iloc[0]['logValue']),
                                   '%0.2f' % (ifg_og.iloc[lenIfg-1]['logValue'])])
        if i == 2:
            plt.ylabel('Ligands with specific pIC50', fontsize=14)
        else:
            plt.ylabel('')
    tk_ndim = ifg.shape[1]
    xlabels = np.arange(0, tk_ndim, tk_ndim//10)
    plt.xticks(xlabels, xlabels)
    plt.xlabel('IFP bits', fontsize=14)

    title = args.IFP_file.replace('.csv', f'_heatmap_IFP')
    plt.savefig(
        os.path.join(args.img_folder, title+'.pdf')
    )
    plt.savefig(
        os.path.join(args.img_folder, title+'.png'),
        dpi=250
    )


def plot_pIC50_bitCounts(df_og, args):
    os.system('mkdir images')
    dfSize = len(df_og)
    step = dfSize//6
    df = df_og.drop(['logValue', 'score_0'], axis=1)
    df['Bits counts'] = df.apply(lambda x: x.sum(), axis=1)
    df = pd.DataFrame(df['Bits counts'])
    df['pIC50'] = df_og['logValue']
    df['Docking Score'] = df_og['score_0']
    color = ['#00008B', '#008B8B']
    types = ['Docking Score', 'Bits counts']
    for type in types:
        fig = plt.figure()
        sns.set(style='ticks')
        # fig = plt.figure(figsize=(7, 4.8))  # default 6.4, 4.8
        plt.rc('font', family='Times New Roman', size=12, weight='bold')

        ifg = df
        paper_rc = {'lines.linewidth': 2, 'lines.markersize': 6}
        sns.set_context("paper", rc=paper_rc)
        sns.scatterplot(x='pIC50', y=type, data=ifg,
                        hue=None, markers=True, alpha=0.3, color=color[types.index(type)])
        sns.kdeplot(x='pIC50', y=type, data=ifg,
                    hue=None, markers=True)
        plt.xlabel('pIC50', fontsize=14)
        plt.ylabel(type, fontsize=14)

        title = f'pIC50_{type}_kde'.strip(" ")
        plt.savefig(
            os.path.join(args.img_folder, title+'.pdf')
        )
        plt.savefig(
            os.path.join(args.img_folder, title+'.png'),
            dpi=250
        )


def main(args):
    analyze_switch = 0
    av_switch = 0
    compress_num = 0
    ifCreateRefer = 0
    ifRemoveAllZeros = 1
    ifp_csv = args.IFP_file
    print(ifp_csv)
    ifp_df = pd.read_csv(ifp_csv)
    print(ifp_df)
    ifp_df = ifp_df.set_index('Molecule')
    if av_switch > 0:
        AV_df = pd.read_csv(args.AV_file)
        AV_df = AV_df.set_index('Molecule')
    if analyze_switch > 0:
        analyze_IFP(ifp_df, args)
    if compress_num > 0:
        top_df = compress_IFP(ifp_df, compress_num, args)
        top_df.to_csv(ifp_csv.replace('.csv', f'_{compress_num}.csv'))

    referName = 'refer_res_nonAll0'
    if ifCreateRefer:
        create_nonzeroRefer(ifp_df, args, referName)
    if ifRemoveAllZeros:
        nonAllZeroDf = remove_allzeros(ifp_df, args, referName)
        print(f'nonAllZeroDf={nonAllZeroDf}')
        nonAllZeroDf.to_csv(args.IFP_file.replace(
            '.csv', '_nonAllZero.csv'), index=True)
    # print(top_df)

    # IFP_AV_df = get_AV(top_df, AV_df)
    # print(IFP_AV_df)
    # heatmap_plot(IFP_AV_df, args)
    # plot_pIC50_bitCounts(IFP_AV_df, args)


if __name__ == "__main__":
    args = get_parser()
    main(args)
