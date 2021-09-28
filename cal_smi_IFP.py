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
#from pathos.multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from pebble import concurrent, ProcessPool
from concurrent.futures import TimeoutError


def smi_pdb(id_smi):
    # chembl_id = col['ChEMBL ID']
    chembl_id = id_smi[0]
    smi = id_smi[1]
    # path=Path(path)
    opfile = f'{chembl_id}.pdb'
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
        # print(mol)
        mol.make3D(forcefield='mmff94', steps=100)
        mol.localopt()
        mol.write(format='pdb', filename=str(opfile), overwrite=True)
        return 1
    except:
        print(f"Tranformation of {smi} failed! ")
        return 0


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def save_obj(obj, name):
    # os.system('mkdir obj')
    with open(name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def prepare_ligand(file, config):
    prepare_ligand4 = config["prepare_ligand4"]
    # SavePath=Path(save_path)
    file = Path(file)
    op_file = f'{file.stem}.pdbqt'
    cmd = f"{prepare_ligand4} -l {file} -o {op_file}"
    try:
        os.system(cmd)
        os.system(f'rm {file}')
        print(f"{file} has been processed successfully! ")
        return 1
    except:
        print(f"{file} has been omitted! ")
        if not os.path.isfile('./docking_failed/'):
            os.system(f'mkdir ./docking_failed/')
        os.system(f'mv {file} ./docking_failed/')
        return 0


def config_create(template, ligand, machine):
    print(f'configcreate_machine={machine}')
    ligand = Path(ligand)
    # job_id = random.randint(1, 999999)
    template_file = open(template, 'r')
    new_config = f'./config_run_{machine}_{ligand.stem}.txt'
    config_file = open(new_config, 'w')
    for line in template_file.readlines():
        line_new = line.replace('$ligand', str(ligand))
        # line_new = line_new.replace('$n_jobs', str(n_jobs))
        config_file.write(line_new)
    return new_config


def run_dock(file_path, save_path, machine, config):
    vina = config["vina_path"]
    print(f'run_dock_machine={machine}')
    if os.path.isfile(file_path):
        os.system(f"mv {file_path} {save_path}/")
        filename = os.path.basename(file_path)
        config_template = './config_vina.txt'
        new_config = config_create(config_template,
                                   f'{save_path}/{filename}', machine)
        outfile = filename.replace('.pdbqt', '_out.pdbqt')
        if not os.path.exists(f'{save_path}/{outfile}'):
            print(f'start processing {save_path}/{filename}')
            cmd = f"{vina} --config {new_config} --log /dev/null"
            print(cmd)
            os.system(cmd)
        else:
            print(f'skipped {filename}')
        os.system(f'rm {new_config}')
        os.system(f'rm {save_path}/{filename}')
        print(f"{filename} has been processed successfully! ")
        return 1
    else:
        return 0


def dock(id_smi, config, save_path='./', machine='LocalPC'):
    smi_pdb(id_smi)
    print(f"id_smi={id_smi}")
    file = f'{id_smi[0]}.pdb'
    prepare_ligand(file, config)
    file = f'{id_smi[0]}.pdbqt'
    print(f'procesing: {id_smi} ')
    print(f'dock_machine={machine}')
    run_dock(file, save_path, machine, config)
    #     # os.system(f'rm {file}')
    #     return 1
    # except Exception as e:
    #     print(f"Something went wrong while processing: {id_smi}")
    #     print(e)
    #     return 0


def find_files(path):
    files = os.listdir(path)
    pdbqt_list = []
    for file in files:
        file_sp = file.split('_')
        if len(file_sp) < 2:
            print(file+'\t was omitted!')
            continue
        elif file_sp[1] == 'out.pdbqt':
            pdbqt_list.append(file)
    print(f"{len(pdbqt_list)} files have been detected!")
    return pdbqt_list


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
    simple_name = base_name.replace('_', '.').split('.')
    simple_name = simple_name[0]

    processed = {'simple_name': simple_name,
                 'base_name': base_name, 'full_name': ligand, 'fullPath': ligand}
    df_interactions, scores = cal_interacions_run(processed, mol_p, config)
    simple_names = [f"{simple_name}_{i}" for i in range(len(scores))]
    reference_atom = load_obj(ifpRefer[0][0])
    reference_res = load_obj(ifpRefer[1][0])
    AAIFPs, RESIFPs = [], []
    for idf_interaction in df_interactions:
        AAIFP, RESIFP = cal_IFP(
            idf_interaction, reference_atom, reference_res)
        AAIFPs.append(AAIFP)
        RESIFPs.append(RESIFP)
    # df_Interaction = concat_df(df_res)
    # colname = []
    # for iatm in reference_atom:
    #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
    #         colname.append(f'{iatm}_{iifp}')
    # # print(colname)
    # # print(AAIFP)
    # df_AAIFP = pd.DataFrame(
    #     AAIFPs, columns=colname)
    # colname = []
    # for ires in reference_res:
    #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
    #         colname.append(f'{ires}_{iifp}')
    # df_RESIFP = pd.DataFrame(
    #     RESIFPs, columns=colname)

    # compress the IFPs
    for ipose in range(len(AAIFPs)):
        if ifpRefer[0][1] != '':
            reference_atom = list(load_obj(ifpRefer[0][1]))
            # print(reference_atom)
            AAIFPs[ipose] = AAIFPs[ipose][reference_atom]
        # AAIFP = list(AAIFP.loc[0])
        # print(AAIFP)
        if ifpRefer[1][1] != '':
            reference_res = list(load_obj(ifpRefer[1][1]))
            RESIFP[ipose] = RESIFP[ipose][reference_res]
        # RESIFP = list(RESIFP.loc[0])
    IFP_list = [simple_names, AAIFPs, RESIFPs, scores]
    # print(f"IFP_list={IFP_list}")
    return IFP_list
    # except Exception as e:
    #     print(e)
    #     return None


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


def plot_bits(df, ImgName):
    # print(IFP_AV_df)
    os.system('mkdir images')
    sns.set(style='ticks')
    plt.figure(figsize=(7, 4.8))
    plt.rc('font', family='Times New Roman', size=12, weight='bold')
    # ifg = df
    ifg = df.copy(deep=True)
    ifg = ifg.drop(['Docking score', 'smi'], axis=1)
    if len(ifg) > 100:
        ifg = ifg.head(100)
        print(ifg.columns)
    ifg = ifg.astype('int')
    paper_rc = {'lines.linewidth': 1, 'lines.markersize': 8}
    sns.set_context("paper", rc=paper_rc)
    sns.heatmap(ifg, cmap='viridis')
    print(ifg)
    tk_ndim = ifg.shape[1]
    print(f"tk_ndim={tk_ndim}")
    xlabels = np.arange(0, tk_ndim, tk_ndim//10)
    plt.xticks(xlabels, xlabels)
    tk_len = ifg.shape[0]
    print(f"tk_len={tk_len}")
    ylabels = np.arange(0, tk_len, tk_len//10)
    ylabels2 = list(ylabels.copy())
    ylabels2[0] = 'Seed'
    plt.yticks(ylabels, ylabels2)
    # g.set(xticklabels=[])
    # plt.yscale('log')
    plt.xlabel('IFP bits', fontsize=14)
    # plt.xlim(0, 200)
    plt.ylabel('Molecules', fontsize=14)
    title = f"Interaction Fingerprint with Seed {ImgName}"
    plt.savefig(
        os.path.join('./images', title+'_heatmap.pdf')
    )
    plt.savefig(
        os.path.join('./images', title+'_heatmap.png'),
        dpi=300)


def cal_molSimilarity(seedSmi, smis, top_num):
    seedMol = Chem.MolFromSmiles(seedSmi)
    valSmi = [smi for smi in smis if Chem.MolFromSmiles(smi) != None]
    print(float(len(valSmi))/len(smis))
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
    tanimoto = float(common)/(sum(IFP1)+sum(IFP2)-common)
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
    # if ifIFP:
    for temp in tempList:
        resList = load_obj(f"{sampledSmi}_{temp}")
        resListNew = []  # store IFP Data
        for resDic in resList:
            # print(f'resDic={resDic}')
            # if resList.index(resDic) > 0:
            #     break  # for fast debug

            # try:  # for robust

            smiList = resDic['smis']
            seedSmi = resDic['seedSmi']
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
                    smiId.append([f'{chemblId}{idx}', col['smi']])
                    dfSmi.loc[idx, 'id'] = f'{chemblId}{idx}'

            tmp_path = f'./dockTmp_{sampledSmi.split("/")[-1]}/{chemblId}_{temp}'
            # dock section
            print('#'*10+'\tDocking start!\t'+'#'*10)
            if ifdock:
                os.system(f'mkdir ./dockTmp_{sampledSmi.split("/")[-1]}')
                # os.system(f'rm -r  {tmp_path}')
                tmpPath_path = Path(tmp_path)
                if not tmpPath_path.exists():
                    os.system(f'mkdir {tmp_path}')
                    # with Pool(args.n_jobs) as pool:
                    #     dock_p = partial(dock, config=config, save_path=str(
                    #         tmp_path), machine=args.machine)
                    #     results = pool.map(dock_p, smiId)
                with ProcessPool(max_workers=args.n_jobs) as pool:
                    # dock_p = partial(dock, save_path=str(tmp_path),
                    #                  machine=args.machine)
                    print("RUNING POOL!!!!")
                    for ismiId in smiId:
                        future = pool.schedule(dock, args=[ismiId, config, str(
                            tmp_path), args.machine], timeout=300)
                    # future = pool.map(
                    #     dock, smiId, save_path=str(tmp_path), timeout=300)

            print(
                '#' * 10 + '\tCalculate interaction fingerprint start!\t' + '#' * 10)
            df_Interaction = {'df_hbond': '', 'df_halogen': '',
                              'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
            protein = config['protein']
            config['ligand_folder'] = tmp_path
            ligand_folder = tmp_path
            ligands = walk_folder(ligand_folder, '_out.pdbqt')
            print(ligands)
            protein = parse_protein_vina(protein)
            mol_p = Molecule(protein['protein'], protein=True)
            if ifIFP:
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
                # resDicNew['AAIFP'] = AAIFP_full
                # resDicNew['ResIFP'] = ResIFP_full
                # resDicNew['dockScore'] = dockScore
                # resDicNew['IFPIDs'] = IFPIDs

                # reference_atom = load_obj('refer_atoms_list')
                # reference_res = load_obj('refer_res_list')
                # colname = []
                # for iatm in reference_atom:
                #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                #         colname.append(f'{iatm}_{iifp}')
                # AAIFP_full = pd.DataFrame(
                #     AAIFP_full, columns=colname)
                # colname = []
                # for ires in reference_res:
                #     for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                #         colname.append(f'{ires}_{iifp}')
                # ResIFP_full = pd.DataFrame(
                #     ResIFP_full, columns=colname)
            if ifSimilarity:
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in AAIFP_full]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in AAIFP_full]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in ResIFP_full]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in ResIFP_full]
                # resDicNew['ifpSim'] = ifpSims
                # resDicNew['bitRec'] = bitRecv
                dfRes = pd.DataFrame(IFPIDs, columns=['id'])
                # print(f'dfRes={dfRes}')
                dfRes['ifpSim'] = ifpSims
                dfRes['bitRec'] = bitRecv
                dfRes['dockScore'] = dockScore
                dfRes['AIFP'] = AAIFP_full
                dfRes['ResIFP'] = ResIFP_full
                dfRes.set_index('id', inplace=True)
                # print(f'dfRes={dfRes}')
                # print(f'dfSmi={dfSmi}')
                dfSmi.set_index('id', inplace=True)
                # dfResIdx = [idx for idx in list(
                #     dfSmi.index) if idx in dfRes.index]
                # dfResIdx = list(set(dfResIdx))

                # print(f'dfRes={dfRes}')
                # print(f'dfResIdx={len(dfResIdx)}')
                # print(f'dfRes.index={len(dfRes.index)}')
                # dfRes = dfRes.loc[dfResIdx]
                # dfRes = pd.DataFrame(dfRes)
                # print(f'dfRes.columns=  {dfRes.columns}')
                # print(f'dfSmi.columns=  {dfSmi.columns}')
                for idx, col in dfRes.iterrows():
                    idx_nopose = idx.split("_")[0]
                    dfRes.loc[idx, 'smi'] = dfSmi.loc[idx_nopose]['smi']
                    dfRes.loc[idx, 'molSim'] = dfSmi.loc[idx_nopose]['similarity']
                # print(dfRes)
                resDicNew['dfRes'] = dfRes

                # print(f'ifpSims: {ifpSims}')
                # print(f'bitRecv: {bitRecv}')
            resListNew.append(resDicNew)

            # except Exception as e:
            #     print(e)
            #     continue
        with open(f'{sampledSmi}_{temp}_ifp' + '.pkl', 'wb') as f:
            pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)
    if ifPlot:
        ifpSimList = []  # store IFP Data
        for temp in tempList:
            resList = load_obj(f"{sampledSmi}_{temp}_ifp")
            for resDic in resList:
                # smiList = resDic['smis']
                seedSmi = resDic['seedSmi']
                chemblId = resDic['molID']
                dfRes = resDic['dfRes']
                ifpSimList.append([seedSmi, f'Seed,chemblID:{chemblId}'])
                for idx, col in dfRes.iterrows():
                    # print(idx)
                    # print(col)
                    id = col.name
                    molSim = col['molSim']
                    ifpSim = col['ifpSim']
                    bitRecv = col['bitRec']
                    dockScore = col['dockScore']

                    ifpSimList.append(
                        [col['smi'], f"id:{id}  molSimilarity: {molSim} ifpSimilarity: {ifpSim}  bitsRecovery: {bitRecv}  dockingScore: {dockScore}"])

            ifpDf = pd.DataFrame(ifpSimList, columns=['SMILES', 'Name'])
            ifpDf.to_csv(f"{sampledSmi}_{temp}.csv", index=None)
        # for figType in ['IFP tanimoto similarity', 'Seed positive bits recovery']:
        #     ifp=ifpDf[ifpDf['type'] == figType]
        #     sns.set(style = 'ticks')
        #     # plt.figure(figsize=(7, 5.4))
        #     sns.displot(data = ifp, x = 'value',
        #                 hue = 'Temperature', kind = 'kde', fill = True, linewidth = 2)
        #     plt.rc('font', family = 'Times New Roman', size = 12, weight = 'bold')
        #     paper_rc={'lines.linewidth': 8, 'lines.markersize': 8}
        #     sns.set_context("paper", rc = paper_rc)
        #     #  markers=True, style='Type')
        #     # g.set(xticklabels=[])
        #     # plt.yscale('log')
        #     plt.xlabel(figType, fontsize = 14)
        #     # plt.xlim(75, 110)
        #     plt.ylabel('Density', fontsize = 14)
        #     title=f'{figType} generated SMILES Epoch 400'
        #     # plt.title(title, fontsize=14)
        #     plt.tight_layout()
        #     # plt.legend('middle left')
        #     plt.savefig(
        #         os.path.join('images', title+'.pdf')
        #     )
        #     plt.savefig(
        #         os.path.join('images', title+'.png'),
        #         dpi= 300
        #     )


def process_IFPReinvent(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    ifIFP = ifCalFP
    ifdock = ifdock
    ifPlot = False
    ifSimilarity = ifSimilarity
    smiFile = open('./reinvent_train_chembl/reinvent_100k.smi', 'r')
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
    tmp_path = './dockTmp_reinvent'
    if ifdock:
        # os.system(f'rm -r  {tmp_path}')
        os.system(f'mkdir {tmp_path}')
        with Pool(args.n_jobs) as pool:
            dock_p = partial(dock, config=config, save_path=str(tmp_path),
                             machine=args.machine)
            results = [x for x in tqdm(
                pool.imap(dock_p, smiId),
                total=len(smiId),
                miniters=10
            )
                if x is not None]
        print(
            '#' * 10 + '\tCalculate interaction fingerprint start!\t' + '#' * 10)
        df_Interaction = {'df_hbond': '', 'df_halogen': '',
                          'df_elecpair': '', 'df_hydrophobic': '', 'df_pistack': ''}
        protein = config['protein']
        config['ligand_folder'] = tmp_path
        ligand_folder = tmp_path
        ligands = walk_folder(ligand_folder, '_out.pdbqt')
        print(ligands)
        protein = parse_protein_vina(protein)
        mol_p = Molecule(protein['protein'], protein=True)
    ifpRefer_org = [['./AIFP/obj/refer_atoms_list', ''],
                    ['./AIFP/obj/refer_res_list', ''], 'atom', 'adjust']
    if ifIFP:
        ligand_folder = tmp_path
        config['ligand_folder'] = tmp_path
        ligands = walk_folder(ligand_folder, '_out.pdbqt')
        with Pool(args.n_jobs) as pool:
            IFP_p = partial(
                IFP, ifpRefer=ifpRefer_org, config=config)
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
            AAIFP_full.append(ires[1])
            IFPIDs.append(ires[0])
            dockScore.append(ires[3])
            ResIFP_full.append(ires[2])
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
            resListNew = []  # store IFP Data
            molSimCombine = []
            ifpSimCombine = []
            bitRecvCombine = []
            smiCombine = smiListReinvent
            dockScoreCombine = dockScore
            for resDic in resList:
                smiList = resDic['smis']
                seedSmi = resDic['seedSmi']
                chemblId = resDic['molID']
                seedIfp = resDic['SeedIFP']
                if ifSimilarity:
                    dfSim = cal_molSimilarity(
                        seedSmi, smiListReinvent, len(smiList))
                    molSimCombine += list(dfSim['similarity'])
                    print(f"molSimCombine\n{molSimCombine}")
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
                    ifpSimCombine += ifpSims
                    bitRecvCombine += bitRecv
            resListNew = {'sampledSmi': smiCombine, "ifpSim": ifpSimCombine, "molSim": molSimCombine,
                          "bitRecv": bitRecvCombine, "dockScore": dockScoreCombine}
            with open(f'{sampledSmi}_{temp}_reinventIfpSim' + '.pkl', 'wb') as f:
                pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_IFPActive(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    # ifIFP = ifCalFP
    # ifdock = ifdock
    # ifPlot = False
    activeFile = '../cdk2_activity/cdk2_pic50.csv'
    activeDf = pd.read_csv(activeFile)
    activeDf = activeDf.set_index("Molecule")
    activeIdx = list(activeDf.index)
    skipLines = 50
    activeIdx = activeIdx[-1000:]
    dockPath = 'AIFP/Data/CDK2_prepared_Results'

    if ifpRefer[2] == 'atom':
        ActiveDf = pd.read_csv('AIFP/cdk2_target_AAIFP_smi.csv')
        if ifpRefer[0][1] != '':
            ActiveDf = pd.read_csv('AIFP/cdk2_target_AAIFP_smi_nonAllZero.csv')
    elif ifpRefer[2] == 'res':
        ActiveDf = pd.read_csv('AIFP/cdk2_target_ResIFP_smi.csv')
        if ifpRefer[1][1] != '':
            ActiveDf = pd.read_csv(
                'AIFP/cdk2_target_ResIFP_smi_nonAllZero.csv')

    # get SMILES
    ActiveDf = ActiveDf.set_index('Molecule')
    activeIdx = [idx for idx in activeIdx if idx in ActiveDf.index]
    ActiveDf = ActiveDf.loc[activeIdx]
    smiListAct = list(ActiveDf['smi'])
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
    dockScore = get_dScore(activeIdx, dockPath)
    for temp in tempList:
        resList = load_obj(f"{sampledSmi}_{temp}")
        resListNew = []  # store IFP Data
        molSimCombine = []
        ifpSimCombine = []
        bitRecvCombine = []
        smiCombine = smiListAct
        dockScoreCombine = dockScore
        for resDic in resList:
            smiList = resDic['smis']
            seedSmi = resDic['seedSmi']
            chemblId = resDic['molID']
            seedIfp = resDic['SeedIFP']
            if ifSimilarity:
                dfSim = cal_molSimilarity(
                    seedSmi, smiListAct, len(smiList))
                molSimCombine += list(dfSim['similarity'])
                print(f"molSimCombine\n{molSimCombine}")
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in IFPList]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in IFPList]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in IFPList]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in IFPList]
                ifpSimCombine += ifpSims
                bitRecvCombine += bitRecv
        resListNew = {'sampledSmi': smiCombine, "ifpSim": ifpSimCombine, "molSim": molSimCombine,
                      "bitRecv": bitRecvCombine, "dockScore": dockScoreCombine}
        with open(f'{sampledSmi}_{temp}_activeIfpSim' + '.pkl', 'wb') as f:
            pickle.dump(resListNew, f, pickle.HIGHEST_PROTOCOL)


def process_randomChembl(sampledSmi, tempList, ifdock, ifCalFP, ifSimilarity, ifpRefer, args, config):
    # ifIFP = ifCalFP
    # ifdock = ifdock
    # ifPlot = False
    dockPath = 'AIFP/Data/ChEMBL27_Results/2'
    skipLines = 50
    if ifpRefer[2] == 'atom':
        chemblDf = pd.read_csv('AIFP/cdk2_2_AAIFP_smi.csv')
        if ifpRefer[0][1] != '':
            chemblDf = pd.read_csv('AIFP/cdk2_2_AAIFP_smi_nonAllZero.csv')
    elif ifpRefer[2] == 'res':
        chemblDf = pd.read_csv('AIFP/cdk2_2_ResIFP_smi.csv')
        if ifpRefer[1][1] != '':
            chemblDf = pd.read_csv('AIFP/cdk2_2_ResIFP_smi_nonAllZero.csv')
    # get SMILES
    chemblDf = chemblDf.sample(2000)
    IdList = list(chemblDf['Molecule'])
    smiListAct = list(chemblDf['smi'])
    # get IFPs
    rmCols = ['Molecule', 'smi']
    for colName in rmCols:
        try:
            chemblDf = chemblDf.drop([colName], axis=1)
        except Exception as e:
            print(e)
            continue
    IFPList = []
    for idx, row in chemblDf.iterrows():
        IFPList.append(list(row))
    # get docking score
    dockScore = get_dScore(IdList, dockPath)
    for temp in tempList:
        resList = load_obj(f"{sampledSmi}_{temp}")
        resListNew = []  # store IFP Data
        molSimCombine = []
        ifpSimCombine = []
        bitRecvCombine = []
        smiCombine = smiListAct
        dockScoreCombine = dockScore
        for resDic in resList:
            smiList = resDic['smis']
            seedSmi = resDic['seedSmi']
            chemblId = resDic['molID']
            seedIfp = resDic['SeedIFP']
            if ifSimilarity:
                dfSim = cal_molSimilarity(
                    seedSmi, smiListAct, len(smiList))
                molSimCombine += list(dfSim['similarity'])
                print(f"molSimCombine\n{molSimCombine}")
                if ifpRefer[2] == 'atom':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in IFPList]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in IFPList]
                if ifpRefer[2] == 'res':
                    ifpSims = [cal_ifpTanimoto(
                        IFPItm, seedIfp, ifpRefer) for IFPItm in IFPList]
                    bitRecv = [cal_bitRecovery(
                        seedIfp, IFPItm) for IFPItm in IFPList]
                ifpSimCombine += ifpSims
                bitRecvCombine += bitRecv
        resListNew = {'sampledSmi': smiCombine, "ifpSim": ifpSimCombine, "molSim": molSimCombine,
                      "bitRecv": bitRecvCombine, "dockScore": dockScoreCombine}
        with open(f'{sampledSmi}_{temp}_randomChemblIfpSim' + '.pkl', 'wb') as f:
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
        ifpRefer = [['./obj/refer_atoms_list', ''],
                    ['./obj/refer_res_list', ''], 'atom', 'adjust']  # 2: res atom 3: adjust full
    # for compressed atom
    if ifpMode == 'atomCompressed':
        ifpRefer = [['./obj/refer_atoms_list', './obj/refer_atoms_nonzero'],
                    ['./obj/refer_res_list', ''], 'atom', 'full']  # 2: res atom 3: adjust full
    # for full res
    if ifpMode == 'res':
        ifpRefer = [['./obj/refer_atoms_list', ''],
                    ['./obj/refer_res_list', ''], 'res', 'adjust']  # 2: res atom 3: adjust full
    # for compressed res
    if ifpMode == 'resCompressed':
        ifpRefer = [['./obj/refer_atoms_list', ''],
                    ['./obj/refer_res_list', './obj/refer_res_nonAll0'], 'res', 'full']  # 2: res atom 3: adjust full
    # tempList = [0.1, 0.2, 0.4, 0.6, 1.0]
    tempList = [1.0]
    sampledSmi = args.sampledSmi
    process_IFP(sampledSmi, tempList, ifdock, ifIFP,
                ifSim, ifpRefer, args, config)

    # ifdock = False

    # process_IFPReinvent(sampledSmi, tempList, ifdock, ifIFP,
    #                     ifSim, ifpRefer, args, config)

    # process_IFPActive(sampledSmi, tempList, ifdock, ifIFP,
    #                   ifSim, ifpRefer, args, config)
    # process_randomChembl(sampledSmi, tempList, ifdock, ifIFP,
    #                      ifSim, ifpRefer, args, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sampledSmi", help="sampled SMILES(pkl)", default='')
    parser.add_argument(
        "--save_path", help="path to save docking results", default='./')
    parser.add_argument("--n_jobs", type=int,
                        help="jobs", default=10)
    parser.add_argument("--machine", type=str,
                        help="machine name", default='')
    parser.add_argument("--ifpMode", type=str,
                        help="mode of IFP", default='')
    parser.add_argument("--config", type=str,
                        help="config of IFP calculation", default='./config_ifp.txt')
    args = parser.parse_args()
    config = parse_config_vina(args.config)
    main(args, config)


# smi_pdb(['chembl_test','CCCCC=O'], './')
# file='./chembl_test.pdb'
# prepare_ligand(file, './')
# docking('./chembl_test.pdbqt','./','47')
