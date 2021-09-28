try:
    from openbabel import pybel
except:
    import pybel
import pandas as pd
import numpy as np
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm
import pickle
import os
import sys
from pathlib import Path
import rdkit
from rdkit import Chem
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from AIFP.create_IFP_batch import walk_folder
from AIFP.model.toolkits.parse_conf import parse_config_vina, parse_protein_vina, parse_ligand_vina
# from model.toolkits.parse_conf import parse_config
from model.obbl import Molecule
from model.toolkits.spatial import angle, distance
from model.toolkits.interactions import hbonds, pi_stacking, salt_bridges, \
    hydrophobic_contacts, close_contacts, halogenbonds
from model.toolkits.pocket import pocket_atoms
from model.IFP import cal_Interactions,  get_Molecules, cal_IFP
# from pathos.multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt
from pebble import concurrent, ProcessPool
from concurrent.futures import TimeoutError
from random import sample

glide = "/mnt/home/zhangjie/Bin/Schrodinger2017/glide"
structconvert = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/structconvert"
prepwizard = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/prepwizard"
mae_toPdbqt = '/mnt/home/zhangjie/Projects/cRNN/AIFP/mae_toPdbqt.py'


def smi_mae(id_smi):
    ''' The SMILES of ligand will be transformed into maestro format.
    '''
    # chembl_id = col['ChEMBL ID']
    chembl_id = id_smi[0]
    smi = id_smi[1]
    # path=Path(path)
    opfile = f'{chembl_id}'
    print(opfile)
    try:
        # smi = col['Smiles']
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

        # print(mol)
        mol.addh()
        mol.title = chembl_id
        # print(mol)
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='mol2', filename=f'{opfile}.mol2', overwrite=True)
        os.system(f'{structconvert} -imol2 {opfile}.mol2 -omae {opfile}.mae')
        #  clean
        if os.path.exists(f'{opfile}.mol2'):
            os.system(f'rm {opfile}.mol2')
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        if os.path.exists(f'{opfile}.mol2'):
            os.system(f'rm {opfile}.mol2')
        return 0


def strip_pose(idx):
    idx_nopose = ''
    idx_sp = idx.split("_")
    for itm in range(len(idx_sp)-1):
        if idx_nopose == '':
            idx_nopose = idx_sp[itm]
        else:
            idx_nopose += f'_{idx_sp[itm]}'
    return idx_nopose


def write_dockInput(id_smi, args):
    dockInput_new_file = f'{id_smi[0]}.in'
    dockInput_new_f = open(dockInput_new_file, 'w')
    with open(args.dockInput_template, 'r') as dockInput_template_f:
        for line in dockInput_template_f.readlines():
            line_new = line.replace('$MaeFile', f'{str(id_smi[0])}.mae')
            # line_new = line_new.replace('$n_jobs', str(n_jobs))
            dockInput_new_f.write(line_new)
    dockInput_template_f.close()
    dockInput_new_f.close()
    return dockInput_new_file


def new_config(old_config, new_config, dock_folder):
    config_new_f = open(new_config, 'w')
    with open(old_config, 'r') as config_template_f:
        for line in config_template_f.readlines():
            line_sp = line.split()
            if len(line_sp) > 1:
                if line_sp[0] == 'ligand_folder':
                    line = f'ligand_folder {dock_folder}\n'
            config_new_f.write(line)
    config_template_f.close()
    config_new_f.close()


def dock(id_smi, args):
    # dock a single compounds
    smi_mae(id_smi)
    dockInput_new_file = write_dockInput(id_smi, args)
    print(f'dockInput_new_f= {dockInput_new_file}')
    os.system(f'{glide} -WAIT -OVERWRITE -NOJOBID  {dockInput_new_file}')
    # clean the output
    tempFiles = [f"{id_smi[0]}.in", f"{id_smi[0]}.mae"]
    for ifile in tempFiles:
        os.system(f'rm {ifile}')


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    # os.system('mkdir obj')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def find_files(path):
    files = os.listdir(path)
    maegz_list = []
    for file in files:
        if 'pv.maegz' not in file:
            print(file+'\t was omitted!')
            continue
        else:
            file = file.strip('_pv.maegz')
            maegz_list.append(file)
    print(f"{len(maegz_list)} files have been detected!")
    return maegz_list


def read_sampledSmi(csv):
    res = []
    idx = 0
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
                    res.append([f'seed', orgSmi])
                else:
                    res.append([f'Ligand{idx}', lineSp[0]])
                    idx += 1
                    if idx > 150:
                        break
            except Exception as e:
                print(e)
    return res


def cal_interacions_run(ligand, mol_p, config):
    '''Calculate the interaction between all the pose and the protein!
    '''
    ligand_name = ligand['simple_name']
    ligand_folder = config['ligand_folder']
    ligand = parse_ligand_vina(os.path.join(
        ligand_folder, ligand['base_name']))
    print(ligand)
    # ligand = parse_ligand_vina(ligand['fullPath'])

    mol_ls = [Molecule(idocked_ligand, protein=False)
              for idocked_ligand in ligand['docked_ligands']]
    df_res = [cal_Interactions(
        mol_p, imol_l, config) for imol_l in mol_ls]
    print(f"\n{ligand_name}\n")
    # for key in df_res.keys():
    #     df_res[key]['Molecule'] = ligand_name
    score = ligand['scorelist']
    return df_res, score
    #     df_Interaction = concat_df(df_Interaction, df_res)


def IFP(ligand, ifpRefer, config):
    # try:
    protein = config['protein']
    protein = parse_protein_vina(protein)
    mol_p = Molecule(protein['protein'], protein=True)
    base_name = os.path.basename(ligand)

    # print(base_name)
    simple_name = base_name.strip('_out.pdbqt')

    processed = {'simple_name': simple_name,
                 'base_name': base_name, 'full_name': ligand, 'fullPath': ligand}
    df_interactions, scores = cal_interacions_run(processed, mol_p, config)
    assert len(df_interactions) == len(
        scores), f'Check the file: {ligand} carefully!'
    simple_names = [f"{simple_name}_pose_{i}" for i in range(len(scores))]
    reference_atom = load_obj(ifpRefer[0][0])
    reference_res = load_obj(ifpRefer[1][0])
    AAIFPs, RESIFPs = [], []
    for ipose in range(len(df_interactions)):
        idf_interaction = df_interactions[ipose]
        AAIFP, RESIFP = cal_IFP(
            idf_interaction, reference_atom, reference_res)
        # AAIFP = [f'{simple_name}_{ipose}']+AAIFP
        # RESIFP = [f'{simple_name}_{ipose}']+RESIFP
        AAIFPs.append(AAIFP)
        RESIFPs.append(RESIFP)

    # compress the IFPs
    # for ipose in range(len(AAIFPs)):
    #     if ifpRefer[0][1] != '':
    #         reference_atom = list(load_obj(ifpRefer[0][1]))
    #         AAIFPs[ipose] = AAIFPs[ipose][reference_atom]
    #     if ifpRefer[1][1] != '':
    #         reference_res = list(load_obj(ifpRefer[1][1]))
    #         RESIFP[ipose] = RESIFP[ipose][reference_res]
    IFP_list = [simple_names, AAIFPs, RESIFPs, scores]
    # print(f"IFP_list={IFP_list}")
    return IFP_list


def remove_zeros(df_ifp):
    '''
    Remove the columns of all zeros
    '''
    # df_ifp = df_ifp.set_index('Molecule')
    # count_col['Molecule'] = ''
    df_bits = df_ifp.copy(deep=True)
    df_bits = df_bits.drop(['Docking score', 'smi'], axis=1)
    count_col = df_bits.astype('int').sum()
    count_col.sort_values(ascending=False, inplace=True)
    pd_count_col = pd.DataFrame(count_col, columns=['counts'])
    pd_count_col = pd_count_col[pd_count_col['counts'] > 0]

    nonzero_AAIFP = df_ifp[list(pd_count_col.index)+['Docking score', 'smi']]
    print(nonzero_AAIFP)
    return nonzero_AAIFP


def cal_molSimilarity(seedSmi, smis, top_num):
    seedMol = Chem.MolFromSmiles(seedSmi)
    valSmi = [smi for smi in smis if Chem.MolFromSmiles(smi) != None]
    # print(float(len(valSmi))/len(smis))
    mols = [Chem.MolFromSmiles(smi)
            for smi in valSmi]
    seedFP = AllChem.GetMorganFingerprintAsBitVect(
        seedMol, 2, nBits=1024)
    FPs = [AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=1024) for mol in mols]
    molSims = [DataStructs.TanimotoSimilarity(
        FP, seedFP) for FP in FPs]
    # molSims = [DataStructs.DiceSimilarity(
    #     FP, seedFP) for FP in FPs]
    print(f"molSims= {molSims}")
    smiSim = []
    for i in range(len(valSmi)):
        smiSim.append([valSmi[i], molSims[i]])
    df = pd.DataFrame(smiSim, columns=['smi', 'similarity'])
    df.sort_values(by='similarity', ascending=False, inplace=True)
    df = df.head(top_num)
    return df


def cal_ifpTanimoto(IFP1, IFP2, ifpRefer):
    print(len(IFP1))
    # print(IFP1)
    print(len(IFP2))
    # print(IFP2)
    IFP1 = [int(i) for i in IFP1]
    IFP2 = [int(i) for i in IFP2]
    assert len(IFP1) == len(IFP2)
    # print(f'IFP1: {len(IFP1)}')
    # print(f'IFP2: {IFP2}')
    lenIFP = len(IFP1)
    common = 0
    for idx in range(lenIFP):
        if float(IFP1[idx]) > 0.5:
            if float(IFP1[idx])-float(IFP2[idx]) < 0.5:
                common += 1
    # if ifpRefer[2] == 'atom':
    #     tanimoto = (common-1425)/((lenIFP-1425)*2-common+1425)
    # elif ifpRefer[2] == 'res':
    #     tanimoto = (common-200)/((lenIFP-200)*2-common+200)
    # if ifpRefer[3] == 'full':
    #     tanimoto = (common - 0) / ((lenIFP - 0) * 2 - common + 0)
    tanimoto = float(common)/(sum(IFP1)+sum(IFP2)-common+0.000001)
    print(f'common: {common}  tanimoto: {tanimoto}')
    return tanimoto


def cal_bitRecovery(IFPref, IFP2):
    assert len(IFPref) == len(IFP2)
    # print(f'IFPref: {len(IFPref)}')
    # print(f'IFP2: {IFP2}')
    lenIFP = len(IFPref)
    common = 0
    oneCounts = 0+1e-8
    for idx in range(lenIFP):
        if float(IFPref[idx]) > 0.5:
            oneCounts += 1.
            if float(IFPref[idx])-float(IFP2[idx]) < 0.5:
                common += 1.

    bitRecovery = common / oneCounts * 100
    print(f'common= {common}  bitRecovery={bitRecovery}')
    return bitRecovery


def get_dScore(IdList, dockPath):
    '''
    get the docking score from the vina output
    '''
    files = walk_folder(dockPath, '_out.pdbqt')
    count = 0
    dScoreList = []
    for file in files:
        filename = file.replace('_out.pdbqt', '')
        if filename in IdList:
            count += 1
            # if count > 10:
            #     break
            print(f'count: {count}')
            outfile = os.path.join(dockPath, file)
            ligand_dic = parse_ligand_vina(outfile)
            score = ligand_dic['scorelist']
            # filename = file.replace('_out.pdbqt', '')
            # print
            # index = df[df['ID'] == filename].index
            dScoreList.append(score[0])
    return dScoreList


def process_IFP(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):

    ifIFP = ifCalFP
    ifdock = ifdock
    ifSimilarity = ifSimilarity
    ifPlot = False
    for temp in tempList:
        resList = load_obj(f"{sampledSmi}_{temp}")
        resListNew = []  # store IFP Data
        for resDic in resList:
            smiList = resDic['smis']
            if len(smiList) < 1:
                '''Empty generated SMILES will be skipped!'''
                continue
            seedSmi = resDic['seedSmi']
            print(f'seedSmi={seedSmi}')
            chemblId = resDic['molID']
            seedIfp = resDic['SeedIFP']
            resDicNew = resDic.copy()
            # choose top 20 similarity for docking
            # len(smiList)  for choose all
            dfSmi = cal_molSimilarity(seedSmi, smiList, len(smiList))
            smiId = []
            for idx, col in dfSmi.iterrows():
                # if idx > 2:
                #     break  # for fast debug
                mol = Chem.MolFromSmiles(col['smi'])
                if mol:
                    atoms = mol.GetAtoms()
                    natm = len(atoms)
                    if natm > 50:
                        continue
                    smiId.append([f'{chemblId}_sampl_{idx}', col['smi']])
                    dfSmi.loc[idx, 'id'] = f'{chemblId}_sampl_{idx}'
            dfSmi.dropna(axis=0, inplace=True, how='any')
            if len(dfSmi) < 1 or 'id' not in dfSmi.columns:
                continue
            dock_path = f'./dockTmp_{sampledSmi.split("/")[-1]}/{chemblId}_{temp}'
            tmp_path = f'{dock_path}/glide'
            # dock section
            print('#'*10+'\tDocking start!\t'+'#'*10)
            if ifdock:
                Path(tmp_path).mkdir(parents=True, exist_ok=True)
                os.chdir(tmp_path)
                ''' Skipped the docked compounds!'''
                docked_list = find_files('./')
                with ProcessPool(max_workers=args.n_jobs) as pool:
                    # dock_p = partial(dock, save_path=str(tmp_path),
                    #                  machine=args.machine)
                    print("RUNING POOL!!!!")
                    for ismiId in smiId:
                        if ismiId[0] not in docked_list:
                            future = pool.schedule(
                                dock, args=[ismiId, args], timeout=300)
                            print(
                                f'{ismiId[0]} has been scheduled for docking!')
                        else:
                            print(
                                f'{ismiId[0]} has been already docked before!')
                    # future = pool.map(
                    #     dock, smiId, save_path=str(tmp_path), timeout=300)
                os.system(
                    f"python {mae_toPdbqt}  --path ../glide --n_jobs={args.n_jobs}")
                os.chdir('../../../')
            if ifIFP:
                tmp_path = f'{dock_path}/glide_pdbqt'
                os.system(f'mkdir {tmp_path}')
                print(
                    '#' * 10 + '\tCalculate interaction fingerprint start!\t' + '#' * 10)
                df_Interaction = {'df_hbond': '', 'df_halogen': '',
                                  'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
                if not Path(tmp_path).exists():
                    continue
                tmp_path = f'{dock_path}/glide_pdbqt'
                new_config(
                    args.config, f'{dock_path}/config_ifp_generate.txt', tmp_path)
                os.system(
                    f'python /mnt/home/zhangjie/Projects/cRNN/AIFP/create_IFP_batch.py --config {dock_path}/config_ifp_generate.txt --n_jobs {args.n_jobs} --save {dock_path}/generate_ifp')
                df_AAIFP = pd.read_csv(f"{dock_path}/generate_ifp_AAIFP.csv")
                df_AAIFP.set_index('Molecule', inplace=True)
                df_RESIFP = pd.read_csv(f"{dock_path}/generate_ifp_ResIFP.csv")
                df_RESIFP.set_index('Molecule', inplace=True)
                dfRes = pd.read_csv(f"{dock_path}/generate_ifp_AAIFP.csv")
                dfRes = dfRes[['Molecule']]
                dfRes.set_index('Molecule', inplace=True)
                AAIFP, RESIFP = [], []
                for idx in dfRes.index:
                    '''Fetching docking score'''
                    print(f'Retrieving docking score of {idx}')
                    idx_sp = idx.split('_')  # split name_poseID
                    pdbqt_file = f'{strip_pose(idx)}_out.pdbqt'
                    ligand_dic = parse_ligand_vina(
                        f'{dock_path}/glide_pdbqt/{pdbqt_file}')
                    score = ligand_dic['scorelist'][int(idx_sp[-1])]
                    dfRes.loc[idx, 'dockScore'] = score
                    '''Fetching AAIFP'''
                    AAIFP.append(list(df_AAIFP.loc[idx]))
                    # dfRes.loc[idx, 'AAIFP'] = AAIFP
                    '''Fetching RESIFP'''
                    RESIFP.append(list(df_RESIFP.loc[idx]))
                    # dfRes.loc[idx, 'RESIFP'] = RESIFP

            if ifSimilarity:
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in AAIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in AAIFP]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in RESIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in RESIFP]
                dfRes['ifpSim'] = ifpSims
                dfRes['bitRec'] = bitRecv
                dfRes['AAIFP'] = AAIFP
                dfRes['RESIFP'] = RESIFP
                dfSmi.set_index('id', inplace=True)

                for idx, col in dfRes.iterrows():
                    try:
                        idx_nopose = strip_pose(idx)
                        dfRes.loc[idx, 'smi'] = dfSmi.loc[idx_nopose]['smi']
                        dfRes.loc[idx,
                                  'molSim'] = dfSmi.loc[idx_nopose]['similarity']
                    except Exception as e:
                        print(e)
                        continue
                # print(dfRes)
                resDicNew['dfRes'] = dfRes
            resListNew.append(resDicNew)

        with open(f'{sampledSmi}_{temp}_ifp' + '.pkl', 'wb') as f:
            pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_IFPReinvent_v1(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    ifIFP = ifCalFP
    ifdock = ifdock
    ifPlot = False
    ifSimilarity = ifSimilarity
    smiFile = open(
        '/mnt/home/zhangjie/Projects/cRNN/CDK2-glide/reinvent_100k.smi', 'r')
    smiListReinvent = []
    for line in smiFile.readlines():
        lineSp = line.split()
        smi = lineSp[0]
        smiListReinvent.append(smi)
        if len(smiListReinvent) > 2000:  # read just 2000 smiles
            break
    smiId = []  # prepare the input for docing
    for idx, smi in enumerate(smiListReinvent):
        smiId.append([f'reinvent{idx}', smi])
    dfSmi = pd.DataFrame(smiId, columns=['id', 'smi'])
    tmp_path = f'./dockTmp_reinvent/glide'
    if ifdock:
        os.system(f'mkdir ./dockTmp_reinvent')
        os.system(f'mkdir {tmp_path}')
        os.chdir(tmp_path)
        ''' Skipped the docked compounds!'''
        docked_list = find_files('./')
        with ProcessPool(max_workers=args.n_jobs) as pool:
            print("RUNING POOL!!!!")
            for ismiId in smiId:
                if ismiId[0] not in docked_list:
                    future = pool.schedule(
                        dock, args=[ismiId, args], timeout=300)
                    print(
                        f'{ismiId[0]} has been scheduled for docking!')
                else:
                    print(
                        f'{ismiId[0]} has been already docked before!')
            # future = pool.map(
            #     dock, smiId, save_path=str(tmp_path), timeout=300)
        os.system(
            f"python {mae_toPdbqt}  --path ../glide --n_jobs={args.n_jobs}")
        os.chdir('../../')
        print(
            '#' * 10 + '\tCalculate interaction fingerprint start!\t' + '#' * 10)
        df_Interaction = {'df_hbond': '', 'df_halogen': '',
                          'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
        # protein = config['protein']
        # protein = parse_protein_vina(protein)
        # mol_p = Molecule(protein['protein'], protein=True)
    # ifpRefer_org = [['./AIFP/obj/refer_atoms_list', ''],
    #                 ['./AIFP/obj/refer_res_list', ''], 'atom', 'adjust']
    if ifIFP:
        tmp_path = f'./dockTmp_reinvent/glide_pdbqt'
        ligand_folder = tmp_path
        config['ligand_folder'] = tmp_path
        ligands = walk_folder(ligand_folder, '_out.pdbqt')
        with Pool(args.n_jobs) as pool:
            IFP_p = partial(
                IFP, ifpRefer=ifpRefer, config=config)
            res_list = [x for x in tqdm(
                pool.imap(IFP_p, list(ligands)),
                total=len(ligands),
                miniters=50
            )
                if x is not None]
        AAIFP_full = []
        ResIFP_full = []
        IFPIDs = []
        dockScore = []
        for ires in res_list:
            AAIFP_full.extend(ires[1])
            IFPIDs.extend(ires[0])
            dockScore.extend(ires[3])
            ResIFP_full.extend(ires[2])
        resDic = {}
        resDic['AAIFP'] = AAIFP_full
        resDic['ResIFP'] = ResIFP_full
        resDic['dockScore'] = dockScore
        resDic['IFPIDs'] = IFPIDs
        save_obj(resDic, f'{sampledSmi}_reinvent_ifp')

    if ifSimilarity:
        reinventDic = load_obj(f'{sampledSmi}_reinvent_ifp')
        reference_atom = load_obj(ifpRefer[0][0])
        reference_res = load_obj(ifpRefer[1][0])
        AAIFP_full = reinventDic['AAIFP']
        ResIFP_full = reinventDic['ResIFP']
        dockScore = reinventDic['dockScore']
        IFPIDs = reinventDic['IFPIDs']
        dfRes = pd.DataFrame(IFPIDs, columns=['id'])
        # dfRes['ifpSim'] = ifpSims
        # dfRes['bitRec'] = bitRecv
        dfRes['dockScore'] = dockScore
        # dfRes['AIFP'] = AAIFP_full
        # dfRes['ResIFP'] = ResIFP_full
        dfRes.set_index('id', inplace=True)
        dfSmi.set_index('id', inplace=True)
        for idx, col in dfRes.iterrows():
            idx_nopose = idx.split("_pose_")[0]
            dfRes.loc[idx, 'smi'] = dfSmi.loc[idx_nopose]['smi']

        colname = []
        for iatm in reference_atom:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{iatm}_{iifp}')
        # print(colname)
        AAIFP = pd.DataFrame(
            AAIFP_full, columns=colname)
        colname = []
        for ires in reference_res:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{ires}_{iifp}')
        RESIFP = pd.DataFrame(
            ResIFP_full, columns=colname)
        # compress the IFPs
        if ifpRefer[0][1] != '':
            reference_atom = list(load_obj(ifpRefer[0][1]))
            print(reference_atom)
            AAIFP = AAIFP[reference_atom]
        # AAIFP = list(AAIFP)
        AAIFPList = []
        for idx, row in AAIFP.iterrows():
            AAIFPList.append(list(row))
        AAIFP = AAIFPList
        # print(AAIFP)
        if ifpRefer[1][1] != '':
            reference_res = list(load_obj(ifpRefer[1][1]))
            RESIFP = RESIFP[reference_res]
        # RESIFP = list(RESIFP)
        RESIFPList = []
        for idx, row in RESIFP.iterrows():
            RESIFPList.append(list(row))
        RESIFP = RESIFPList
        for temp in tempList:
            resList = load_obj(f"{sampledSmi}_{temp}")
            resListNew = []  # store result Data
            for resDic in resList:
                dfRes_temp = dfRes.copy(deep=True)
                # invalid_idx = []
                # for idx, row in dfRes_temp.iterrows():
                #     smi = row['smi']
                #     if Chem.MolFromSmiles(smi) == None:
                #         invalid_idx.append(idx)
                # dfRes_temp = dfRes_temp.drop(invalid_idx, axis=0)
                smiList = resDic['smis']
                seedSmi = resDic['seedSmi']
                chemblId = resDic['molID']
                seedIfp = resDic['SeedIFP']
                '''There are some SMILES which is valid in openbabel but is not valid in rdkit.'''
                seedMol = Chem.MolFromSmiles(seedSmi)
                seedFP = AllChem.GetMorganFingerprintAsBitVect(
                    seedMol, 2, nBits=1024)
                for idx, row in dfRes_temp.iterrows():
                    try:
                        ismi = row['smi']
                        imol = Chem.MolFromSmiles(ismi)
                        iFP = AllChem.GetMorganFingerprintAsBitVect(
                            imol, 2, nBits=1024)
                        imolSim = DataStructs.TanimotoSimilarity(
                            iFP, seedFP)
                        dfRes_temp.loc[idx, 'molSim'] = imolSim
                    except Exception as e:
                        print(e)
                        continue
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in AAIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in AAIFP]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in RESIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in RESIFP]
                dfRes_temp['ifpSim'] = ifpSims
                dfRes_temp['bitRec'] = bitRecv
                dfRes_temp = dfRes_temp.dropna(axis=0)
            resListNew.append({'dfRes': dfRes_temp})
            with open(f'{sampledSmi}_{temp}_reinventIfpSim' + '.pkl', 'wb') as f:
                pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_IFPActive(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    '''Get the IFPs from previous ifp results.'''
    # ifIFP = ifCalFP
    # ifdock = ifdock
    # ifPlot = False
    activeFile = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Data/a2a_kiLt20nm.csv'
    activeDf = pd.read_csv(activeFile)
    activeDf = activeDf.set_index("ChEMBL ID")
    activeIdx = list(activeDf.index)

    # skipLines = 50
    # activeIdx = activeIdx[-1000:]
    dockPath = '../Data/a2a_active_dock/0_pdbqt'
    dscore_df = pd.read_csv(
        '../AIFP_files/a2aactive_AAIFP_dScorePP.csv')

    if ifpRefer[2] == 'atom':
        ActiveDf = pd.read_csv(
            '../AIFP_files/a2aactive_AAIFP.csv')
    #     if ifpRefer[0][1] != '':
    #         ActiveDf = pd.read_csv(
    #             './AIFP_files/cdk2_target_AAIFP_smi_nonAllZero.csv')
    # elif ifpRefer[2] == 'res':
    #     ActiveDf = pd.read_csv(
    #         '../AIFP_files/cdk2_active_ResIFP_AIFPsmi_5pose.csv')
    #     if ifpRefer[1][1] != '':
    #         ActiveDf = pd.read_csv(
    #             './AIFP_files/cdk2_target_ResIFP_smi_nonAllZero.csv')

    # get SMILES
    ActiveDf['index'] = ActiveDf['Molecule']
    ActiveDf = ActiveDf.set_index('index')
    activeIdx_pose = []
    for idx in activeIdx:
        for ipose in range(10):
            if f"{idx}_{ipose}" in ActiveDf.index:
                activeIdx_pose.append(f"{idx}_{ipose}")
    # activeIdx = [idx for idx in activeIdx if idx in ActiveDf.index]

    ActiveDf = ActiveDf.loc[activeIdx_pose]
    dscore_df['index'] = dscore_df['Molecule']
    dscore_df = dscore_df.set_index('index')
    dscore_df = dscore_df.loc[activeIdx_pose]

    dfRes_temp = dscore_df[['Molecule', 'smi']].copy(deep=True)

    dfRes_temp['index'] = dfRes_temp['Molecule']
    dfRes_temp = dfRes_temp.set_index('index')
    dfRes_temp['dockScore'] = dscore_df.loc[dfRes_temp.index]['score_0']
    smiListAct = list(dscore_df['smi'])
    # get IFPs
    rmCols = ['Molecule', 'smi']
    for colName in rmCols:
        try:
            ActiveDf = ActiveDf.drop([colName], axis=1)
        except Exception as e:
            print(e)
            continue
    IFPList = []
    for idx, row in ActiveDf.iterrows():
        IFPList.append(list(row))
    # get docking score
    # dockScore = get_dScore(activeIdx, dockPath)

    for temp in tempList:
        resList = load_obj(f"{sampledSmi}_{temp}")
        resListNew = []  # store result Data
        for resDic in resList:
            dfRes_temp = dfRes_temp.copy(deep=True)
            smiList = resDic['smis']
            seedSmi = resDic['seedSmi']
            chemblId = resDic['molID']
            seedIfp = resDic['SeedIFP']
            '''There are some SMILES which is valid in openbabel but is not valid in rdkit.'''
            seedMol = Chem.MolFromSmiles(seedSmi)
            seedFP = AllChem.GetMorganFingerprintAsBitVect(
                seedMol, 2, nBits=1024)
            for idx, row in dfRes_temp.iterrows():
                try:
                    ismi = row['smi']
                    imol = Chem.MolFromSmiles(ismi)
                    iFP = AllChem.GetMorganFingerprintAsBitVect(
                        imol, 2, nBits=1024)
                    imolSim = DataStructs.TanimotoSimilarity(
                        iFP, seedFP)
                    dfRes_temp.loc[idx, 'molSim'] = imolSim
                except Exception as e:
                    print(e)
                    continue
            # if ifpRefer[2] == 'atom':
            #     ifpSims = [cal_ifpTanimoto(
            #         IFPItm, seedIfp, ifpRefer) for IFPItm in AAIFP]
            #     bitRecv = [cal_bitRecovery(
            #         seedIfp, IFPItm) for IFPItm in AAIFP]
            # if ifpRefer[2] == 'res':
            #     ifpSims = [cal_ifpTanimoto(
            #         IFPItm, seedIfp, ifpRefer) for IFPItm in RESIFP]
            #     bitRecv = [cal_bitRecovery(
            #         seedIfp, IFPItm) for IFPItm in RESIFP]
            ifpSims = [cal_ifpTanimoto(
                IFPItm, seedIfp, ifpRefer) for IFPItm in IFPList]
            bitRecv = [cal_bitRecovery(
                seedIfp, IFPItm) for IFPItm in IFPList]
            dfRes_temp['ifpSim'] = ifpSims
            dfRes_temp['bitRec'] = bitRecv
            dfRes_temp = dfRes_temp.dropna(axis=0)
        resListNew.append({'dfRes': dfRes_temp})
        with open(f'{sampledSmi}_{temp}_activeIfpSim' + '.pkl', 'wb') as f:
            pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_randomChembl(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    dscore_df = pd.read_csv(
        '../AIFP_files/a2achembl0_AAIFP_dScorePP.csv')
    dscore_df['index'] = dscore_df['Molecule']
    dscore_df = dscore_df.set_index('index')
    if ifpRefer[2] == 'atom':
        chemblDf = pd.read_csv(
            '../AIFP_files/a2achembl0_AAIFP.csv')
    #     if ifpRefer[0][1] != '':
    #         chemblDf = pd.read_csv('AIFP/cdk2_2_AAIFP_smi_nonAllZero.csv')
    # elif ifpRefer[2] == 'res':
    #     chemblDf = pd.read_csv(
    #         '../AIFP_files/cdk2_chembl2_ResIFP_AIFPsmi_5pose.csv')
    #     if ifpRefer[1][1] != '':
    #         chemblDf = pd.read_csv('AIFP/cdk2_2_ResIFP_smi_nonAllZero.csv')
    # get SMILES
    chemblDf['index'] = chemblDf['Molecule']
    chemblDf = chemblDf.set_index('index')
    index_df = chemblDf.index
    index_nopose = [strip_pose(idx) for idx in index_df]
    random1k_index = sample(index_nopose, 1000)
    random1k_pose = []
    # chemblDf = chemblDf.sample(1000)
    for idx in random1k_index:
        for ipose in range(10):
            if f"{idx}_{ipose}" in chemblDf.index:
                random1k_pose.append(f"{idx}_{ipose}")
    chemblDf = chemblDf.loc[random1k_pose]
    dscore_df = dscore_df.loc[random1k_pose]
    dfRes_temp = dscore_df[['Molecule', 'smi', 'score_0']].copy(deep=True)
    dfRes_temp['index'] = dfRes_temp['Molecule']
    # dfRes_temp = dfRes_temp.set_index('index')
    # dfRes_temp['dockScore'] = dscore_df.loc[dfRes_temp.index]['score_0']
    # get IFPs
    rmCols = ['index', 'smi', 'Molecule', 'logP', 'QED', 'SA',
              'Wt', 'NP', 'score_0', 'TPSA', 'MW', 'HBA', 'HBD', 'QED']
    for colName in rmCols:
        try:
            dscore_df = dscore_df.drop([colName], axis=1)
        except Exception as e:
            print(e)
            continue
    IFPList = []
    for idx, row in dscore_df.iterrows():
        IFPList.append(list(row))
    # get docking score

    for temp in tempList:
        resList = load_obj(f"{sampledSmi}_{temp}")
        resListNew = []  # store result Data
        for resDic in resList:
            dfRes_temp = dfRes_temp.copy(deep=True)
            smiList = resDic['smis']
            seedSmi = resDic['seedSmi']
            chemblId = resDic['molID']
            seedIfp = resDic['SeedIFP']
            '''There are some SMILES which is valid in openbabel but is not valid in rdkit.'''
            seedMol = Chem.MolFromSmiles(seedSmi)
            seedFP = AllChem.GetMorganFingerprintAsBitVect(
                seedMol, 2, nBits=1024)
            for idx, row in dfRes_temp.iterrows():
                try:
                    ismi = row['smi']
                    imol = Chem.MolFromSmiles(ismi)
                    iFP = AllChem.GetMorganFingerprintAsBitVect(
                        imol, 2, nBits=1024)
                    imolSim = DataStructs.TanimotoSimilarity(
                        iFP, seedFP)
                    dfRes_temp.loc[idx, 'molSim'] = imolSim
                except Exception as e:
                    print(e)
                    continue
            ifpSims = [cal_ifpTanimoto(
                IFPItm, seedIfp, ifpRefer) for IFPItm in IFPList]
            bitRecv = [cal_bitRecovery(
                seedIfp, IFPItm) for IFPItm in IFPList]
            dfRes_temp['ifpSim'] = ifpSims
            dfRes_temp['bitRec'] = bitRecv
            dfRes_temp = dfRes_temp.dropna(axis=0)
        resListNew.append({'dfRes': dfRes_temp})
        with open(f'{sampledSmi}_{temp}_randomChemblIfpSim' + '.pkl', 'wb') as f:
            pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_IFP_test_dock(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    ifIFP = ifCalFP
    ifdock = False
    ifPlot = False
    ifSimilarity = ifSimilarity
    df_seed = pd.read_csv(
        '/mnt/home/zhangjie/Projects/cRNN/AIFP/Data/cdk2_ic50_st50nm.csv')
    # smiListReinvent = list(df_seed['smi'])
    # smiFile = open(
    #     '/mnt/home/zhangjie/Projects/cRNN/CDK2-glide/reinvent_100k.smi', 'r')
    # smiListReinvent = []
    # for line in smiFile.readlines():
    #     lineSp = line.split()
    #     smi = lineSp[0]
    #     smiListReinvent.append(smi)
    #     if len(smiListReinvent) > 2000:  # read just 2000 smiles
    #         break
    smiId = []  # prepare the input for docing
    for idx, row in df_seed.iterrows():
        smiId.append([row['molecule_chembl_id'], row['smi']])
    dfSmi = pd.DataFrame(smiId, columns=['id', 'smi'])
    dfSmi = dfSmi.drop_duplicates(subset=['id'])
    print(dfSmi)
    # sys.exit()
    tmp_path = f'./dockTmp_test/glide'
    if ifdock:
        os.system(f'mkdir ./dockTmp_test')
        os.system(f'mkdir {tmp_path}')
        os.chdir(tmp_path)
        ''' Skipped the docked compounds!'''
        docked_list = find_files('./')
        with ProcessPool(max_workers=args.n_jobs) as pool:
            print("RUNING POOL!!!!")
            for ismiId in smiId:
                if ismiId[0] not in docked_list:
                    future = pool.schedule(
                        dock, args=[ismiId, args], timeout=300)
                    print(
                        f'{ismiId[0]} has been scheduled for docking!')
                else:
                    print(
                        f'{ismiId[0]} has been already docked before!')
            # future = pool.map(
            #     dock, smiId, save_path=str(tmp_path), timeout=300)
        os.system(
            f"python {mae_toPdbqt}  --path ../glide --n_jobs={args.n_jobs}")
        os.chdir('../../')
        print(
            '#' * 10 + '\tCalculate interaction fingerprint start!\t' + '#' * 10)
        df_Interaction = {'df_hbond': '', 'df_halogen': '',
                          'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
    if ifIFP:
        tmp_path = f'./dockTmp_test/glide_pdbqt'
        ligand_folder = tmp_path
        config['ligand_folder'] = tmp_path
        ligands = walk_folder(ligand_folder, '_out.pdbqt')
        ligands = ligands[:10]
        with Pool(args.n_jobs) as pool:
            IFP_p = partial(
                IFP, ifpRefer=ifpRefer, config=config)
            res_list = [x for x in tqdm(
                pool.imap(IFP_p, list(ligands)),
                total=len(ligands),
                miniters=50
            )
                if x is not None]
        AAIFP_full = []
        ResIFP_full = []
        IFPIDs = []
        dockScore = []
        for IFP_poses in res_list:
            for ires in range(len(IFP_poses)):
                IFPIDs.append(IFP_poses[0][ires])
                AAIFP_full.append(IFP_poses[1][ires])
                ResIFP_full.append(IFP_poses[2][ires])
                dockScore.append(IFP_poses[3][ires])
        resDic = {}
        resDic['AAIFP'] = AAIFP_full
        resDic['ResIFP'] = ResIFP_full
        resDic['dockScore'] = dockScore
        resDic['IFPIDs'] = IFPIDs
        save_obj(resDic, f'{sampledSmi}_test_ifp')

    if ifSimilarity:
        reinventDic = load_obj(f'{sampledSmi}_test_ifp')
        reference_atom = load_obj(ifpRefer[0][0])
        reference_res = load_obj(ifpRefer[1][0])
        # AAIFP_full = reinventDic['AAIFP']
        # ResIFP_full = reinventDic['ResIFP']
        dockScore = reinventDic['dockScore']
        IFPIDs = reinventDic['IFPIDs']
        dfRes = pd.DataFrame(IFPIDs, columns=['id'])
        dfRes['dockScore'] = dockScore
        dfRes['AIFP'] = AAIFP_full
        dfRes['ResIFP'] = ResIFP_full
        dfRes.set_index('id', inplace=True)
        dfSmi.set_index('id', inplace=True)
        for idx in list(dfRes.index):
            idx_nopose = idx.split("_pose_")[0]
            dfRes.loc[idx, 'smi'] = dfSmi.loc[idx_nopose, 'smi']
        colname = []
        for iatm in reference_atom:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{iatm}_{iifp}')
        AAIFP = AAIFP_full
        RESIFP = ResIFP_full
        for temp in tempList:
            # break
            resList = load_obj(f"{sampledSmi}_{temp}")
            resListNew = []  # store result Data
            for resDic in resList:
                dfRes_temp = dfRes.copy(deep=True)
                # invalid_idx = []
                # for idx, row in dfRes_temp.iterrows():
                #     smi = row['smi']
                #     if Chem.MolFromSmiles(smi) == None:
                #         invalid_idx.append(idx)
                # dfRes_temp = dfRes_temp.drop(invalid_idx, axis=0)
                smiList = resDic['smis']
                seedSmi = resDic['seedSmi']
                chemblId = resDic['molID']
                seedIfp = resDic['SeedIFP']
                '''There are some SMILES which is valid in openbabel but is not valid in rdkit.'''
                seedMol = Chem.MolFromSmiles(seedSmi)
                seedFP = AllChem.GetMorganFingerprintAsBitVect(
                    seedMol, 2, nBits=1024)
                for idx, row in dfRes_temp.iterrows():
                    try:
                        ismi = row['smi']
                        imol = Chem.MolFromSmiles(ismi)
                        iFP = AllChem.GetMorganFingerprintAsBitVect(
                            imol, 2, nBits=1024)
                        imolSim = DataStructs.TanimotoSimilarity(
                            iFP, seedFP)
                        dfRes_temp.loc[idx, 'molSim'] = imolSim
                    except Exception as e:
                        print(e)
                        continue
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in AAIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in AAIFP]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in RESIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in RESIFP]
                dfRes_temp['ifpSim'] = ifpSims
                dfRes_temp['bitRec'] = bitRecv
                dfRes_temp = dfRes_temp.dropna(axis=0)
                resListNew.append({'dfRes': dfRes_temp})
            with open(f'{sampledSmi}_{temp}_testIfpSim' + '.pkl', 'wb') as f:
                pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_IFPReinvent(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    ifIFP = ifCalFP
    ifdock = ifdock
    ifPlot = False
    ifSimilarity = ifSimilarity
    smiFile = open(
        '/mnt/home/zhangjie/Projects/cRNN/CDK2-glide/reinvent_100k.smi', 'r')
    smiListReinvent = []
    for line in smiFile.readlines():
        lineSp = line.split()
        smi = lineSp[0]
        smiListReinvent.append(smi)
        if len(smiListReinvent) > 2000:  # read just 2000 smiles
            break
    smiId = []  # prepare the input for docing
    for idx, smi in enumerate(smiListReinvent):
        smiId.append([f'reinvent{idx}', smi])
    dfSmi = pd.DataFrame(smiId, columns=['id', 'smi'])
    dock_path = './dockTmp_reinvent'
    tmp_path = f'{dock_path}/glide'
    if ifdock:
        os.system(f'mkdir {dock_path}')
        os.system(f'mkdir {tmp_path}')
        os.chdir(tmp_path)
        ''' Skipped the docked compounds!'''
        docked_list = find_files('./')
        with ProcessPool(max_workers=args.n_jobs) as pool:
            print("RUNING POOL!!!!")
            for ismiId in smiId:
                if ismiId[0] not in docked_list:
                    future = pool.schedule(
                        dock, args=[ismiId, args], timeout=300)
                    print(
                        f'{ismiId[0]} has been scheduled for docking!')
                else:
                    print(
                        f'{ismiId[0]} has been already docked before!')
            # future = pool.map(
            #     dock, smiId, save_path=str(tmp_path), timeout=300)
        os.system(
            f"python {mae_toPdbqt}  --path ../glide --n_jobs={args.n_jobs}")
        os.chdir('../../')
        print(
            '#' * 10 + '\tCalculate interaction fingerprint start!\t' + '#' * 10)
        df_Interaction = {'df_hbond': '', 'df_halogen': '',
                          'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
    if ifIFP:
        tmp_path = f'{dock_path}/glide_pdbqt'
        new_config(
            args.config, f'{dock_path}/config_ifp_reinvent.txt', tmp_path)
        os.system(
            f'python /mnt/home/zhangjie/Projects/cRNN/AIFP/create_IFP_batch.py --config {dock_path}/config_ifp_reinvent.txt --n_jobs {args.n_jobs} --save {dock_path}/reinvent_ifp')
        # sys.exit(0)
        df_AAIFP = pd.read_csv(f"{dock_path}/reinvent_ifp_AAIFP.csv")
        df_AAIFP.set_index('Molecule', inplace=True)
        df_RESIFP = pd.read_csv(f"{dock_path}/reinvent_ifp_ResIFP.csv")
        df_RESIFP.set_index('Molecule', inplace=True)
        dfRes = pd.read_csv(f"{dock_path}/reinvent_ifp_AAIFP.csv")
        dfRes = dfRes[['Molecule']]
        dfRes.set_index('Molecule', inplace=True)
        AAIFP, RESIFP = [], []
        for idx in dfRes.index:
            '''Fetching docking score'''
            print(f'Retrieving docking score of {idx}')
            idx_sp = idx.split('_')  # split name_poseID
            pdbqt_file = f'{idx_sp[0]}_out.pdbqt'
            ligand_dic = parse_ligand_vina(
                f'{dock_path}/glide_pdbqt/{pdbqt_file}')
            score = ligand_dic['scorelist'][int(idx_sp[1])]
            dfRes.loc[idx, 'dockScore'] = score
            '''Fetching AAIFP'''
            AAIFP.append(list(df_AAIFP.loc[idx]))
            # dfRes.loc[idx, 'AAIFP'] = AAIFP
            '''Fetching RESIFP'''
            RESIFP.append(list(df_RESIFP.loc[idx]))
            # dfRes.loc[idx, 'RESIFP'] = RESIFP
    if ifSimilarity:
        '''Fetching SMILES'''
        dfSmi.set_index('id', inplace=True)
        for idx in list(dfRes.index):
            idx_nopose = idx.split("_")[0]
            dfRes.loc[idx, 'smi'] = dfSmi.loc[idx_nopose, 'smi']
        # colname = []
        # for iatm in reference_atom:
        #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
        #         colname.append(f'{iatm}_{iifp}')
        # AAIFP = dfRes['AAIFP']
        # RESIFP = dfRes['RESIFP']
        for temp in tempList:
            # break
            resList = load_obj(f"{sampledSmi}_{temp}")
            resListNew = []  # store result Data
            for resDic in resList:
                dfRes_temp = dfRes.copy(deep=True)
                # invalid_idx = []
                # for idx, row in dfRes_temp.iterrows():
                #     smi = row['smi']
                #     if Chem.MolFromSmiles(smi) == None:
                #         invalid_idx.append(idx)
                # dfRes_temp = dfRes_temp.drop(invalid_idx, axis=0)
                smiList = resDic['smis']
                seedSmi = resDic['seedSmi']
                chemblId = resDic['molID']
                seedIfp = resDic['SeedIFP']
                '''There are some SMILES which is valid in openbabel but is not valid in rdkit.'''
                seedMol = Chem.MolFromSmiles(seedSmi)
                seedFP = AllChem.GetMorganFingerprintAsBitVect(
                    seedMol, 2, nBits=1024)
                for idx, row in dfRes_temp.iterrows():
                    try:
                        ismi = row['smi']
                        imol = Chem.MolFromSmiles(ismi)
                        iFP = AllChem.GetMorganFingerprintAsBitVect(
                            imol, 2, nBits=1024)
                        imolSim = DataStructs.TanimotoSimilarity(
                            iFP, seedFP)
                        dfRes_temp.loc[idx, 'molSim'] = imolSim
                    except Exception as e:
                        print(e)
                        continue
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in AAIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in AAIFP]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in RESIFP]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in RESIFP]
                dfRes_temp['ifpSim'] = ifpSims
                dfRes_temp['bitRec'] = bitRecv
                dfRes_temp = dfRes_temp.dropna(axis=0)
                resListNew.append({'dfRes': dfRes_temp})
            with open(f'{sampledSmi}_{temp}_reinventIfpSim' + '.pkl', 'wb') as f:
                pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def main(args, config):
    # debug control
    ifdock = False
    ifIFP = True
    ifSim = ifIFP

    ifpMode = args.ifpMode  # res atom resCompressed atomCompressed
    # control ifp reference and type
    # ['refer_atoms_list', 'refer_atoms_nonzero']  0: original 1: compressed
    # for full atom
    if ifpMode == 'atom':
        ifpRefer = [['../obj/refer_atoms_list', ''],
                    ['../obj/refer_res_list', ''], 'atom', 'adjust']  # 2: res atom 3: adjust full
    # for compressed atom
    if ifpMode == 'atomCompressed':
        ifpRefer = [['../obj/refer_atoms_list', '../obj/refer_atoms_nonzero'],
                    ['../obj/refer_res_list', ''], 'atom', 'full']  # 2: res atom 3: adjust full
    # for full res
    if ifpMode == 'res':
        ifpRefer = [['../obj/refer_atoms_list', ''],
                    ['../obj/refer_res_list', ''], 'res', 'adjust']  # 2: res atom 3: adjust full
    # for compressed res
    if ifpMode == 'resCompressed':
        ifpRefer = [['../obj/refer_atoms_list', ''],
                    ['../obj/refer_res_list', '../obj/refer_res_nonAll0'], 'res', 'full']  # 2: res atom 3: adjust full

    sampledSmi = args.sampledSmi
    # tempList = [0.2, 0.5, 1.0]
    # # tempList = [1.0]
    # process_IFP(sampledSmi, tempList, ifdock, ifIFP,
    #             ifSim, ifpRefer, args, config)

    # ifdock = False
    tempList = [1.0]
    # process_IFPReinvent(sampledSmi, tempList, ifdock, ifIFP,
    #                     ifSim, ifpRefer, args, config)

    # process_IFPActive(sampledSmi, tempList, ifdock, ifIFP,
    #                   ifSim, ifpRefer, args, config)
    process_randomChembl(sampledSmi, tempList, ifdock, ifIFP,
                         ifSim, ifpRefer, args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampledSmi", help="sampled SMILES(pkl)", default='')
    parser.add_argument(
        "--save_path", help="path to save docking results", default='./')
    parser.add_argument("--n_jobs", type=int,
                        help="jobs", default=10)
    parser.add_argument("--ifpMode", type=str,
                        help="mode of IFP", default='')
    parser.add_argument("--config", type=str,
                        help="config of IFP calculation", default='./config_ifp.txt')
    parser.add_argument("--dockInput_template", type=str,
                        help="template of docking template", default='')

    args = parser.parse_args()
    config = parse_config_vina(args.config)
    main(args, config)


# smi_pdb(['chembl_test','CCCCC=O'], './')
# file='./chembl_test.pdb'
# prepare_ligand(file, './')
# docking('./chembl_test.pdbqt','./','47')
