
# %%
from __future__ import print_function
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
import pickle
import pandas as pd
from AIFP.model.obbl import Molecule
from pathlib import Path


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def show_seed(df):
    seed_num = len(df)
    print(f'There are {seed_num} seeds in the results.')
    for i in range(seed_num):
        print(
            f'Index: {i}; pdbId: {df[i]["molID"]}; seedSmi: {df[i]["seedSmi"]}')


def IFP_cols(df_ref):
    colname = []
    for iatm in df_ref:
        for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
            colname.append(f'{iatm}_{iifp}')
    return colname


# %%
'''This is configuration section of this script!!!'''
atomRef_file = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/obj/refer_atoms_list'
df_ref = load_obj(atomRef_file)
resultsFile = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Results/dScorePP_1.0_ifp'
df = load_obj(resultsFile)
protein_file = '/mnt/home/zhangjie/Projects/cRNN/A2A-glide/Data/4UG2_protein_prepared.pdbqt'
convert = ob.OBConversion()
convert.SetInFormat("pdbqt")
protein = ob.OBMol()
convert.ReadFile(protein, protein_file)
protein_mol = Molecule(protein, protein=True)
IFPCols = IFP_cols(df_ref)
# %%


def split_interaction(IFPCols, IFPList, protein_mol):
    def detect_interaction(Bits, IFP_type):
        # select certain type of non-zero IFP bits
        print(f'\nThe {IFP_type} bits as below:')
        counts = 0
        bits_type = []
        for iBits in Bits:
            iBits_sp1 = iBits["bitName"].replace(';', '_').strip().split('_')
            if iBits_sp1[1] == IFP_type:
                counts += 1
                bits_type.append(iBits)
                print(f'{counts}: {iBits}')
        return bits_type
    assert len(IFPCols) == len(IFPList)
    atom_dict = protein_mol.atom_dict
    nonZeroBits = []
    for i in range(len(IFPList)):
        if int(IFPList[i]) == 1:

            value = atom_dict[int(IFPCols[i].split('_')[0])]
            bits_dic = {"bitName": IFPCols[i],
                        "bitIndex": {i},
                        "atomPdbId": f"{value['resname']}_{value['resnum']}_{value['atomtype']}"}
            nonZeroBits.append(
                bits_dic)
    print(f'Non-Zero Bits are: ')
    IFPbit_split = {}
    for IFP_type in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
        IFPbit_split[IFP_type] = detect_interaction(nonZeroBits, IFP_type)
    return IFPbit_split
# Show all the seeds adopted in the sampling.
# show_seed(df)


# Show the interactions between seeds and the protein.
nonZeroBits_Seed_Dic = {}
for iseed in range(len(df)):
    print('\n\n'+'#'*30)
    print(f"Seed pdbID: {df[iseed]['molID']}; Smi: {df[iseed]['seedSmi']}")

    nonZeroBits_Seed = split_interaction(
        IFPCols, df[iseed]['SeedIFP'], protein_mol)
    nonZeroBits_Seed["seed_pdbID"] = df[iseed]['molID']
    nonZeroBits_Seed["seed_smi"] = df[iseed]['seedSmi']
    nonZeroBits_Seed_Dic[df[iseed]['molID']] = nonZeroBits_Seed

# %%
print(df[0]["dfRes"])
# %%
'''Sort sample smiles via IFP score. The weights of different types of interaction ['hbd', 'halg', 'elec', 'elec', 'pipi'] can be assigned with wt'''
wt = {'hbd': 1.0, 'halg': 1.0, 'elec': 0.5, 'hrdr': 0.1, 'pipi': 0.1}
wt_molsim = {'ifp': 0.5, 'molsim': 0.5}


def sortSmi(wt, df):

    for iseed in range(len(df)):

        dfRes = df[iseed]["dfRes"]
        # IFPIDs = df[iseed]["IFPIDs"]
        # print(f"IFPIDs: {IFPIDs}")
        # AAIFP = df[iseed]["dfRes"]['AAIFP']
        molID = df[iseed]['molID']
        print(f"Processing {molID}")
        nonZeroBits = nonZeroBits_Seed_Dic[molID]
        IFP_score_list = []
        # print(dfRes.columns)
        for idx, row in dfRes.iterrows():
            iIFP = row['AAIFP']
            IFPscore = 0
            for IFP_type in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                # print(f"Processing {IFP_type}")
                if len(nonZeroBits[IFP_type]) > 0:
                    bitRec_count = 0
                    for ibit in nonZeroBits[IFP_type]:
                        # print(int(list(ibit["bitIndex"])[0]))
                        if int(iIFP[int(list(ibit["bitIndex"])[0])]) == 1:
                            bitRec_count += 1
                    IFPscore += bitRec_count / \
                        float(len(nonZeroBits[IFP_type]))*wt[IFP_type]
            IFP_score_list.append(IFPscore)
        print(f'IFP_score_list: {IFP_score_list}')
        # sortedSmi_dic = {'molID': df[iseed]
        #                  ["IFPIDs"], 'smis': df[iseed]["smis"], 'dockScore': df[iseed]["dockScore"], 'ifpSim': df[iseed]["ifpSim"], 'bitRec': df[iseed]["bitRec"], 'IFP_Score': IFP_score_list}

        dfRes['IFP_Score'] = IFP_score_list
        dfRes['final_score'] = dfRes['IFP_Score'] * \
            wt_molsim['ifp']+dfRes['molSim']*wt_molsim['molsim']
        # print(f"dfRes={dfRes}")

        # dfRes['IFPscore'] = IFP_score_list
        resultsPath = Path(resultsFile)
        resultsPath = resultsPath.parent.joinpath('sorted_IFPscore')
        Path.mkdir(resultsPath, exist_ok=True)
        sortedSmi_df = dfRes.sort_values(
            by='final_score', ascending=False)
        sortedSmi_df = sortedSmi_df[['ifpSim', 'bitRec',
                                    'dockScore', 'smi', 'molSim', 'IFP_Score', 'final_score']]
        sortedSmi_df.to_csv(
            str(resultsPath.joinpath(f'{molID}_sorted.csv')))


sortSmi(wt, df)
# %%
# print(df[0].keys())
print(df[0]['dfRes'].keys())

# %%
