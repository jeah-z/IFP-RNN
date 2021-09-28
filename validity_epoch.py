from __future__ import print_function
# import ddc.ddc_pub.ddc_v3 as ddc
import pandas as pd
import argparse
import numpy as np
import pickle
import rdkit
from rdkit import Chem
# from ddc.ddc_pub import ddc_v3 as ddc
# from ddc_pub import ddc_v3 as ddc
import numpy as np
import rdkit
import os
from rdkit import Chem
import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
# import h5py
# import ast
import pickle
import glob


def save_obj(obj, name):
    os.system('mkdir obj')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
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
    for smi in smiList:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid += 1
    valid_rate = valid/float(total)*100
    return valid_rate


def plot_valid(df, image_title):

    sns.set(style='ticks')
    plt.figure(figsize=(7, 4.8))
    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    ifg = df
    paper_rc = {'lines.linewidth': 2, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    sns.lineplot(x='epoch', y='validity', data=ifg,
                 hue='Temperature', markers=True, legend='full')
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('Epoch', fontsize=14)
    # plt.xlim(0, 600)
    plt.ylabel('Validity', fontsize=14)
    # logPath = Path(args.log)
    # logName = logPath.name
    title = 'Validity of generated SMILES'
    # plt.title(title, fontsize=14)
    plt.savefig(
        os.path.join('', image_title+'.pdf')
    )
    plt.savefig(
        os.path.join('', image_title+'.png'),
        dpi=250
    )


def prepare_input(IFP_Df, seed_Df):
    inputList = []
    IFP_Df['index'] = IFP_Df['Molecule']
    IFP_Df = IFP_Df.set_index('index')
    for idx, row in seed_Df.iterrows():
        smi = row['smi']
        molID = row['Molecule']
        row = IFP_Df.loc[molID]
        row = row.drop(['smi', 'Molecule'])
        row = np.array(row)
        # row=add_random(row)
        inputDic = {'smi': smi, 'molID': molID, 'row': row}
        inputList.append(inputDic)
        print(f'smi: {smi} molID: {molID} row: {row}')
    return inputList


def find_savedModels(model_path):
    model_parent_path = model_path.parent
    model_name = model_path.stem
    model_files = glob.glob(f'{model_parent_path}/{model_name}--*.zip')
    model_files = [x.replace('.zip', '') for x in model_files]
    return model_files


def write_list(listname, op):
    op.writelines('\tIFP: [')
    for itm in listname:
        op.writelines(f'{itm} ')
    op.writelines('] \n')


def main(args):
    '''A code to calculate the validity of sampled SMILES at different temperature each epoch!'''
    Ifsample = False
    Ifplot = True
    if Ifsample:
        IFP_Df = pd.read_csv(args.IFP_file)
        seed_Df = pd.read_csv(args.seed_csv)
        inputList = prepare_input(IFP_Df, seed_Df)
        model_path = Path(args.model)
        model_files = find_savedModels(model_path)
        print(model_files)
        # os.system(f'mkdir {args.save}')
        validList = []
        validOp = open(f"{args.model}_validity_epoch_log.txt", 'a')
        validOp.writelines('epoch, validity, Temperature\n')
        for epoch in range(10, 500, 10):
            model_epoch = [
                imodel for imodel in model_files if f'--{epoch}--' in imodel]
            if len(model_epoch) != 1:
                print(
                    f"There are more than one file detected! Details as : {model_epoch}")
                break
            model_name = model_epoch[0]
            print(f'model_name= {model_name}')
            try:
                model = ddc.DDC(model_name=model_name)
                for tempValue in [0.0, 0.2, 0.5, 1]:
                    smiList = []
                    for inputDic in inputList:
                        smi = inputDic['smi']
                        molID = inputDic['molID']
                        IFP = inputDic['row']
                        # IFP = np.array(IFP)
                        IFP = np.array([IFP]*128)
                        print(IFP)
                        print(
                            f'Index: {inputList.index(inputDic)}, Sampling for molecule: {molID}')
                        model.batch_input_length = 128
                        smiles, _ = model.predict_batch(
                            latent=IFP, temp=tempValue)
                        smiList += list(smiles)
                        # print(smiles)
                    validity = cal_valid(smiList)
                    print(
                        f"epoch: {epoch}  validity: {validity} Temperature: {tempValue}")
                    validList.append([epoch, validity, str(tempValue)])
                    validOp.writelines(f'{epoch},{validity},{tempValue}\n')
                    validOp.flush()
            except Exception as e:
                print(e)
                continue
        df_valid = pd.DataFrame(
            validList, columns=['epoch', 'validity', 'Temperature'])
        df_valid.to_csv(f"{args.model}_validity_epoch.csv")
    if Ifplot:
        df_valid = pd.read_csv(f"{args.model}_validity_epoch_log.txt")
        df_valid.columns = ['epoch', 'validity', 'Temperature']
        df_valid['epoch'] = df_valid['epoch'].astype('str')
        df_valid = df_valid[df_valid.epoch.apply(lambda x: x.isnumeric())]
        df_valid = df_valid.round(2)
        # df_valid = pd.DataFrame(df_valid)
        print(df_valid)
        df_valid['epoch'] = df_valid['epoch'].astype('int')
        df_valid['validity'] = df_valid['validity'].astype('float')
        df_valid['Temperature'] = df_valid['Temperature'].astype('str')
        df_valid = df_valid[df_valid['Temperature']
                            != '0.0']  # remove temprature 0.0
        plot_valid(df_valid, f"{args.model}_validity_epoch")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--IFP_file", help="IFP file with seed IFP information", type=str,
                        default='')
    parser.add_argument("--model", help="trained model: ./1pose/fullbits.zip", type=str,
                        default='')
    parser.add_argument("--seed_csv", help="csv file with seed name information", type=str,
                        default='')
    # parser.add_argument("--save", help="csv file name to save the results",
    #                     type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
