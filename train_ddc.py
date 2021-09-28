from __future__ import print_function
from ddc.ddc_pub import ddc_v3 as ddc
# import ddc.ddc_pub.ddc_v3 as ddc
import pandas as pd
import argparse
import numpy as np
import pickle
import rdkit
# from openbabel import pybel
try:
    from openbabel import pybel
except:
    import pybel
from rdkit import Chem
from pathlib import Path

import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", help="train file",
                        default='')
    parser.add_argument("--trainDic", help="pickle file for training",
                        default='')
    parser.add_argument("--load_pkl", help="If load training data from pkl file", type=int,
                        default=0)
    parser.add_argument("--save", help="folder to save models",
                        default='./saved_model/')
    args = parser.parse_args()
    return args


def save_obj(obj, name):
    os.system('mkdir obj')
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def charset_update(charset, smi):
    for char in smi:
        if char not in charset:
            charset += char
    return charset


def add_random(bits):
    bits = np.array(bits)
    rands = np.random.rand(len(bits))/10
    bits = bits+rands
    bits[bits < 0.5] = 0
    return bits


def process_csv(df, args):
    # df = pd.read_csv(csv)
    smiList = []
    charset = ''
    molList = []
    bitList = []
    maxlen = 0
    for idx, row in df.iterrows():
        # if idx > 10:  # for debug control
        #     break
        smi = row['smi']
        try:
            # mol = Chem.MolFromSmiles(smi)
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
            smi_clean = mol.write('smi')

            smi_clean = smi_clean.replace('\n', '')
            smi_clean = smi_clean.split()[0]
            print(f'smi_clean: {smi_clean}')
            mol_clean = Chem.MolFromSmiles(smi_clean)
        except Exception as e:
            print(e)
            continue
        if idx % 1000 == 0:
            print(f"Processing csv: Row {idx}")
        row = row.drop(['smi', 'Molecule'])
        # row = add_random(row)
        row = np.array(row)
        charset = charset_update(charset, smi)
        molList.append(mol_clean)
        smiList.append(smi_clean)
        bitList.append(row)
        smiLen = len(smi)
        if smiLen > maxlen:
            maxlen = smiLen
    assert len(bitList) == len(molList)
    print(len(smiList))
    print(len(molList))
    print(charset)
    print(maxlen)
    print(len(bitList))
    trainDic = {'molList': molList, 'bitList': bitList,
                'charset': charset, 'maxlen': maxlen, 'smiList': smiList}
    save_obj(trainDic, args.trainDic)
    return molList, bitList, charset, maxlen, smiList


def main(args):
    save_dir=Path(args.save).parent
    save_dir.mkdir(parents=True,exist_ok=True)
    
    if args.load_pkl <= 0:
        print("Processing csv!")
        fullDf = pd.DataFrame([])
        for csvFile in args.train_csv.split(','):
            print(f"Loading {csvFile}")
            csvFile = pd.read_csv(csvFile)
            if len(fullDf) == 0:
                fullDf = csvFile
            else:
                fullDf = pd.concat([fullDf, csvFile])
        molList, bitList, charset, maxlen, smiList = process_csv(
            fullDf, args)
    elif args.trainDic != '':
        trainDic = load_obj(args.trainDic)
        molList = trainDic['molList']
        bitList = trainDic['bitList']
        charset = trainDic['charset']
        maxlen = trainDic['maxlen']

    dataset_info = {"maxlen": maxlen+50,
                    "charset": charset, "name": "fullBits"}
    IFPmodel = ddc.DDC(
        x=bitList, y=molList, dataset_info=dataset_info, scaling=True)
    IFPmodel.fit(epochs=500, lr=0.001, mini_epochs=1, model_name='fullBits', gpus=1, patience=1, checkpoint_dir=args.save,
                 save_period=10, lr_decay=True, sch_epoch_to_start=200, sch_last_epoch=1000, sch_lr_init=1e-3, sch_lr_final=1e-5)


if __name__ == "__main__":
    args = get_parser()
    main(args)
