from __future__ import print_function
from model.toolkits.interactions import close_contacts
from model.toolkits.interactions import hydrophobic_contacts
from model.toolkits.interactions import salt_bridges
import numpy as np
import pandas as pd
from bitarray import bitarray
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
import sys
import argparse
from time import time
from model.toolkits.parse_conf import parse_config_vina, parse_protein_vina, parse_ligand_vina
# from model.toolkits.parse_conf import parse_config
# from model.toolkits.parse_docking_conf import parse_vina_conf
from model.toolkits.PARAMETERS import HYDROPHOBIC, AROMATIC, HBOND, ELECTROSTATIC, HBOND_ANGLE,\
    AROMATIC_ANGLE_LOW, AROMATIC_ANGLE_HIGH
from model.obbl import Molecule
from model.toolkits.spatial import angle, distance
from model.toolkits.interactions import hbonds, pi_stacking, salt_bridges, \
    hydrophobic_contacts, close_contacts, halogenbonds
from model.toolkits.pocket import pocket_atoms


def create_empty_list(n):
    list_2d = []
    for i in range(n):
        list_2d.append([])
    return list_2d


def list_append2d(list, line):
    size = len(list)
    assert len(line) == size
    for i in range(size):
        list[i].append(line[i])
    return list


def cal_Interactions(mol_p, mol_l, config):
    '''
    The different type interacitons  between specific pose of ligand and protein will be calculated.  The dictionary of interactions will be outputed as as dictionary.
    '''
    # mol_l = Molecule(OBMol=ligand)
    # mol_p = Molecule(OBMol=protein)

    # Hbonds detection
    print('\n'+'#'*10+'\tHbonds detection\t'+'#'*10)
    ad_list, da_list, strict_list = hbonds(
        mol_p, mol_l, cutoff=float(config['hbond_cutoff']), tolerance=120)
    Hbonds = []  # create_empty_list(12)
    for i in range(len(ad_list)):
        if strict_list[i] == False:
            continue
        d = distance([ad_list[i]['coords']], [da_list[i]['coords']])
        new_line = [ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
                    ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13], '', ad_list[i][11], ad_list[i][10]]
        Hbonds.append(new_line)  # = list_append2d(Hbonds, new_line)

        print("Id: %s %s\t AtomicNum: %s %s\t Distance: %s \t IsDonor: %s %s\t IsAcceptor: %s %s\t" %
              (ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
               ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13]))
    df_hbond = pd.DataFrame(
        Hbonds, columns=['Id_p', 'Id_l', 'AtomicNum_p', 'AtomicNum_l', 'Distance',
                         'IsDonor_p', 'IsDonor_l', 'IsAcceptor_p', 'IsAcceptor_l', 'Molecule', 'ResName_p', 'ResNum_p'])

    # Halogenbonds detection
    print('\n'+'#'*10+'\tHalogenbonds detection\t'+'#'*10)
    ad_list, da_list, strict_list = halogenbonds(
        mol_p, mol_l, cutoff=float(config['halogenbond_cutoff']))
    halogen = []  # create_empty_list(12)
    for i in range(len(ad_list)):
        d = distance([ad_list[i]['coords']], [da_list[i]['coords']])
        new_line = [ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
                    ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13], '', ad_list[i][11], ad_list[i][10]]
        halogen.append(new_line)  # = list_append2d(halogen, new_line)
        print("Id: %s %s\t AtomicNum: %s %s\t Distance: %s \t IsDonor: %s %s\t IsAcceptor: %s %s\t" %
              (ad_list[i][0], da_list[i][0], ad_list[i][4], da_list[i][4], round(d[0][0], 2),
               ad_list[i][14], da_list[i][14], ad_list[i][13], da_list[i][13]))
    df_halogen = pd.DataFrame(
        halogen, columns=['Id_p', 'Id_l', 'AtomicNum_p', 'AtomicNum_l', 'Distance',
                          'IsDonor_p', 'IsDonor_l', 'IsAcceptor_p', 'IsAcceptor_l', 'Molecule', 'ResName_p', 'ResNum_p'])

    # Electrostatic detection
    print('\n'+'#'*10+'\tElectrostatic detection\t'+'#'*10)
    elecpair = []  # create_empty_list(12)
    plus_minus, minus_plus = salt_bridges(
        mol_p, mol_l, cutoff=float(config['electrostatic_cutoff']))
    for i in range(len(plus_minus)):
        d = distance([plus_minus[i]['coords']], [minus_plus[i]['coords']])
        new_line = [plus_minus[i][0], minus_plus[i][0], plus_minus[i][4], minus_plus[i][4], round(d[0][0], 2),
                    plus_minus[i][20], minus_plus[i][20], plus_minus[i][19], minus_plus[i][19], '', plus_minus[i][11], plus_minus[i][10]]
        elecpair.append(new_line)  # = list_append2d(elecpair, new_line)
        print("Id: %s %s\t AtomicNum: %s %s\t Distance: %s \t IsPositive: %s %s\t IsNegative: %s %s\t" %
              (plus_minus[i][0], minus_plus[i][0], plus_minus[i][4], minus_plus[i][4], round(d[0][0], 2),
               plus_minus[i][20], minus_plus[i][20], plus_minus[i][19], minus_plus[i][19]))
    df_elecpair = pd.DataFrame(
        elecpair, columns=['Id_p', 'Id_l', 'AtomicNum_p', 'AtomicNum_l', 'Distance',
                           'IsPlus_p', 'IsPlus_l', 'IsMinus_p', 'IsMinus_l', 'Molecule', 'ResName_p', 'ResNum_p'])

    # Hydrophobic detection
    print('\n'+'#'*10+'\tHydrophobic detection\t'+'#'*10)
    h1, h2 = hydrophobic_contacts(
        mol_p, mol_l, cutoff=float(config['hydrophobic_cutoff']))
    hydrophobic = []  # create_empty_list(10)
    for i in range(len(h1)):
        d = distance([h1[i]['coords']], [h2[i]['coords']])
        new_line = [h1[i][0], h2[i][0], h1[i][4], h2[i][4], round(d[0][0], 2),
                    h1[i][17], h2[i][17], '', h1[i][11], h1[i][10]]
        hydrophobic.append(new_line)  # = list_append2d(hydrophobic, new_line)
        print("Id: %s %s\t AtomicNum: %s %s\t Distance: %s \t IsHyrdrophobic: %s %s\t" %
              (h1[i][0], h2[i][0], h1[i][4], h2[i][4], round(d[0][0], 2),
               h1[i][17], h2[i][17]))
    df_hydrophobic = pd.DataFrame(
        hydrophobic, columns=['Id_p', 'Id_l', 'AtomicNum_p', 'AtomicNum_l', 'Distance',
                              'IsHyrdrophobic_p', 'IsHyrdrophobic_l', 'Molecule', 'ResName_p', 'ResNum_p'])

    # Pi_stacking detection
    print('\n'+'#'*10+'\tPi_stacking detection\t'+'#'*10)
    pistack = []  # create_empty_list(10)
    r1, r2, strict_parallel, strict_perpendicular = pi_stacking(
        mol_p, mol_l, cutoff=float(config['pistack_cutoff']))
    for i in range(len(r1)):
        d = distance([r1[i]['centroid']], [r2[i]['centroid']])
        new_line = [r1[i][2], r2[i][2], r1[i][3], r2[i][3], round(d[0][0], 2),
                    r1[i][4], r2[i][4], r1[i][7], r2[i][7], '']
        pistack.append(new_line)  # = list_append2d(pistack, new_line)
        print("ResId: %s %s\t Resnum: %s %s\t Distance: %s \t Resname: %s %s\t Atoms: %s %s\t" %
              (r1[i][2], r2[i][2], r1[i][3], r2[i][3], round(d[0][0], 2),
               r1[i][4], r2[i][4], r1[i][7], r2[i][7]))
    df_pistack = pd.DataFrame(
        pistack, columns=['ResName_p', 'ResName_l', 'ResNum_p', 'ResNum_l', 'Distance', 'ResName_p', 'ResName_l',
                          'AtomList_p', 'AtomList_l', 'Molecule'])

    return {'df_hbond': df_hbond, 'df_halogen': df_halogen, 'df_elecpair': df_elecpair, 'df_hydrophobic': df_hydrophobic, 'df_pistack': df_pistack}


def get_Molecules(df_Interaction):
    molecule_list = []
    for key in df_Interaction.keys():
        if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic', 'df_pistack']:
            interaction = df_Interaction[key]
            molecule_list += list(interaction['Molecule'])
    molecule_list = list(set(molecule_list))
    print(
        f'There are {len(molecule_list)} molecules to be processed!')
    return molecule_list


def reference_atom(df_Interaction):
    protein_atoms = []
    for key in df_Interaction.keys():
        if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic']:
            interaction = df_Interaction[key]
            protein_atoms += list(interaction['Id_p'])
        if key in ['df_pistack']:
            interaction = df_Interaction[key]
            for idx, row in interaction.iterrows():
                for jdx in row['AtomList_p']:
                    protein_atoms.append(jdx)
    protein_atoms = list(set([x for x in protein_atoms if str(x) != '-1']))
    protein_atoms.sort()
    print(
        f'There are {len(protein_atoms)} atoms of protein in the reference list!')
    return protein_atoms


def cal_IFP(df_Interaction, reference_atom, reference_res):
    ''' This function will construction the interaction fingerprint based on the interaction dictionary and references. The results will be returned as two lists, which are atom-based and residue-based separately.
    '''
    atoms_notinrefer = []
    res_notinrefer = []
    AAIFP = [[0, 0, 0, 0, 0] for i in reference_atom]
    RESIFP = [[0, 0, 0, 0, 0] for i in reference_res]
    ifp_types = ['df_hbond', 'df_halogen',
                 'df_elecpair', 'df_hydrophobic', 'df_pistack']
    for key in df_Interaction.keys():
        if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic']:
            interaction = df_Interaction[key]
            # interaction = interaction[interaction['Molecule'] == mole]
            for idx, row in interaction.iterrows():
                # update atom based IFP
                res_NameNum = f"{row['ResName_p']}_{row['ResNum_p']}"
                if row['Id_p'] in reference_atom:
                    ref_idx = reference_atom.index(row['Id_p'])
                    ifp_idx = ifp_types.index(key)
                    AAIFP[ref_idx][ifp_idx] = 1
                else:
                    atoms_notinrefer.append(row['Id_p'])
                # update res based IFP
                if res_NameNum in reference_res:
                    ref_idx = reference_res.index(res_NameNum)
                    ifp_idx = ifp_types.index(key)
                    RESIFP[ref_idx][ifp_idx] = 1
                else:
                    res_notinrefer.append(res_NameNum)

        if key in ['df_pistack']:
            interaction = df_Interaction[key]
            # interaction = interaction[interaction['Molecule'] == mole]
            for idx, row in interaction.iterrows():
                res_NameNum = f"{row['ResName_p']}_{row['ResNum_p']}"
                # update atom based IFP ()
                for jdx in row['AtomList_p']:
                    try:
                        if jdx in AAIFP['AtomID']:
                            ref_idx = reference_atom.index(jdx)
                            ifp_idx = ifp_types.index(key)
                            AAIFP[ref_idx][ifp_idx] = 1
                        else:
                            atoms_notinrefer.append(jdx)

                    except:
                        continue
                # update res based IFP (PiPi)
                if res_NameNum in reference_res:
                    ref_idx = reference_res.index(res_NameNum)
                    ifp_idx = ifp_types.index(key)
                    RESIFP[ref_idx][ifp_idx] = 1
                else:
                    res_notinrefer.append(res_NameNum)
    RESIFP = [str(i) for item in RESIFP for i in item]
    AAIFP = [str(i) for item in AAIFP for i in item]
    atoms_notinrefer = list(set(atoms_notinrefer))
    print("\nAtoms that have interactions however not included in reference:")
    print(f"\nNumber of Atoms: {len(atoms_notinrefer)}\n")
    print(atoms_notinrefer)

    print("\nReses that have interactions however not included in reference:\n")
    res_notinrefer = list(set(res_notinrefer))
    print(f"\nNumber of Reses: {len(res_notinrefer)}\n")
    print(res_notinrefer)

    return AAIFP, RESIFP


class AAIFP_class:
    def __init__(self, df_Interaction, reference_atom, reference_res):
        self.df_Interaction = df_Interaction
        self.reference_atom = reference_atom
        self.reference_res = reference_res
        self.ifp_types = ['df_hbond', 'df_halogen',
                          'df_elecpair', 'df_hydrophobic', 'df_pistack']
        self.AAIFP_full = []
        self.RESIFP_full = []

    def get_Molecules(self):
        _molecule_list = []
        for key in self.df_Interaction.keys():
            if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic', 'df_pistack']:
                interaction = self.df_Interaction[key]
                _molecule_list += list(interaction['Molecule'])
        self.molecule_list = list(set(_molecule_list))
        print(
            f'There are {len(self.molecule_list)} molecules to be processed!')

    def calIFP(self):
        self.get_Molecules()
        _AAIFP_full = []
        _RESIFP_full = []
        atoms_notinrefer = []
        res_notinrefer = []
        for mole in self.molecule_list:
            AAIFP = [[0, 0, 0, 0, 0] for i in self.reference_atom]
            RESIFP = [[0, 0, 0, 0, 0] for i in self.reference_res]
            for key in self.df_Interaction.keys():
                if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic']:

                    interaction = self.df_Interaction[key]
                    interaction = interaction[interaction['Molecule'] == mole]
                    for idx, row in interaction.iterrows():
                        # update atom based IFP
                        res_NameNum = f"{row['ResName_p']}_{row['ResNum_p']}"
                        if row['Id_p'] in self.reference_atom:
                            ref_idx = self.reference_atom.index(row['Id_p'])
                            ifp_idx = self.ifp_types.index(key)
                            AAIFP[ref_idx][ifp_idx] = 1
                        else:
                            atoms_notinrefer.append(row['Id_p'])
                        # update res based IFP
                        if res_NameNum in self.reference_res:
                            ref_idx = self.reference_res.index(res_NameNum)
                            ifp_idx = self.ifp_types.index(key)
                            RESIFP[ref_idx][ifp_idx] = 1
                        else:
                            res_notinrefer.append(res_NameNum)

                if key in ['df_pistack']:
                    interaction = self.df_Interaction[key]
                    interaction = interaction[interaction['Molecule'] == mole]
                    for idx, row in interaction.iterrows():
                        res_NameNum = f"{row['ResName_p']}_{row['ResNum_p']}"
                        # update atom based IFP ()
                        for jdx in row['AtomList_p']:
                            try:
                                if jdx in AAIFP['AtomID']:
                                    ref_idx = self.reference_atom.index(jdx)
                                    ifp_idx = self.ifp_types.index(key)
                                    AAIFP[ref_idx][ifp_idx] = 1
                                else:
                                    atoms_notinrefer.append(jdx)

                            except:
                                continue
                        # update res based IFP (PiPi)
                        if res_NameNum in self.reference_res:
                            ref_idx = self.reference_res.index(res_NameNum)
                            ifp_idx = self.ifp_types.index(key)
                            RESIFP[ref_idx][ifp_idx] = 1
                        else:
                            res_notinrefer.append(res_NameNum)
            RESIFP = [str(i) for item in RESIFP for i in item]
            AAIFP = [str(i) for item in AAIFP for i in item]
            _AAIFP_full.append(AAIFP)
            _RESIFP_full.append(RESIFP)

        atoms_notinrefer = list(set(atoms_notinrefer))
        print("\nAtoms that have interactions however not included in reference:")
        print(f"\nNumber of Atoms: {len(atoms_notinrefer)}\n")
        print(atoms_notinrefer)

        print("\nReses that have interactions however not included in reference:\n")

        res_notinrefer = list(set(res_notinrefer))
        print(f"\nNumber of Reses: {len(res_notinrefer)}\n")
        print(res_notinrefer)

        colname = []
        for iatm in self.reference_atom:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{iatm}_{iifp}')

        self.AAIFP_full = pd.DataFrame(
            _AAIFP_full, columns=colname, index=self.molecule_list)
        colname = []
        for ires in self.reference_res:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{ires}_{iifp}')
        self.RESIFP_full = pd.DataFrame(
            _RESIFP_full, columns=colname, index=self.molecule_list)

        return self.AAIFP_full, self.RESIFP_full

    def compress_IFP(self, num):
        '''
        Compress IFP to remove not important descriptors 
        '''
        count_col = self.AAIFP_full.astype('int').sum()
        count_col.sort_values(ascending=False, inplace=True)
        pd_count_col = pd.DataFrame(count_col, columns=['counts'])
        pd_count_col.to_csv('Interaction_occurrence_count.csv')
        top_col = count_col[range(num)]
        top_AAIFP = self.AAIFP_full[top_col.index]
        return top_AAIFP

    def remove_allzeros(self):
        '''
        Remove the columns of all zeros 
        '''
        count_col = self.AAIFP_full.astype('int').sum()
        count_col.sort_values(ascending=False, inplace=True)
        pd_count_col = pd.DataFrame(count_col, columns=['counts'])
        pd_count_col = pd_count_col[pd_count_col['counts'] > 0]

        nonzero_AAIFP = self.AAIFP_full[pd_count_col.index]
        return nonzero_AAIFP


class AAIFP_batch:
    def __init__(self, df_Interaction, reference_atom, reference_res):
        self.df_Interaction = df_Interaction
        self.reference_atom = reference_atom
        self.reference_res = reference_res
        self.ifp_types = ['df_hbond', 'df_halogen',
                          'df_elecpair', 'df_hydrophobic', 'df_pistack']
        self.AAIFP_full = []
        self.RESIFP_full = []

    def get_Molecules(self):
        _molecule_list = []
        for key in self.df_Interaction.keys():
            if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic', 'df_pistack']:
                interaction = self.df_Interaction[key]
                _molecule_list += list(interaction['Molecule'])
        self.molecule_list = list(set(_molecule_list))
        print(
            f'There are {len(self.molecule_list)} molecules to be processed!')

    def calIFP(self):
        self.get_Molecules()
        _AAIFP_full = []
        _RESIFP_full = []
        atoms_notinrefer = []
        res_notinrefer = []
        for mole in self.molecule_list:
            AAIFP = [[0, 0, 0, 0, 0] for i in self.reference_atom]
            RESIFP = [[0, 0, 0, 0, 0] for i in self.reference_res]
            for key in self.df_Interaction.keys():
                if key in ['df_hbond', 'df_halogen', 'df_elecpair', 'df_hydrophobic']:

                    interaction = self.df_Interaction[key]
                    interaction = interaction[interaction['Molecule'] == mole]
                    for idx, row in interaction.iterrows():
                        # update atom based IFP
                        res_NameNum = f"{row['ResName_p']}_{row['ResNum_p']}"
                        if row['Id_p'] in self.reference_atom:
                            ref_idx = self.reference_atom.index(row['Id_p'])
                            ifp_idx = self.ifp_types.index(key)
                            AAIFP[ref_idx][ifp_idx] = 1
                        else:
                            atoms_notinrefer.append(row['Id_p'])
                        # update res based IFP
                        if res_NameNum in self.reference_res:
                            ref_idx = self.reference_res.index(res_NameNum)
                            ifp_idx = self.ifp_types.index(key)
                            RESIFP[ref_idx][ifp_idx] = 1
                        else:
                            res_notinrefer.append(res_NameNum)

                if key in ['df_pistack']:
                    interaction = self.df_Interaction[key]
                    interaction = interaction[interaction['Molecule'] == mole]
                    for idx, row in interaction.iterrows():
                        res_NameNum = f"{row['ResName_p']}_{row['ResNum_p']}"
                        # update atom based IFP ()
                        for jdx in row['AtomList_p']:
                            try:
                                if jdx in AAIFP['AtomID']:
                                    ref_idx = self.reference_atom.index(jdx)
                                    ifp_idx = self.ifp_types.index(key)
                                    AAIFP[ref_idx][ifp_idx] = 1
                                else:
                                    atoms_notinrefer.append(jdx)

                            except:
                                continue
                        # update res based IFP (PiPi)
                        if res_NameNum in self.reference_res:
                            ref_idx = self.reference_res.index(res_NameNum)
                            ifp_idx = self.ifp_types.index(key)
                            RESIFP[ref_idx][ifp_idx] = 1
                        else:
                            res_notinrefer.append(res_NameNum)
            RESIFP = [str(i) for item in RESIFP for i in item]
            AAIFP = [str(i) for item in AAIFP for i in item]
            _AAIFP_full.append(AAIFP)
            _RESIFP_full.append(RESIFP)

        atoms_notinrefer = list(set(atoms_notinrefer))
        print("\nAtoms that have interactions however not included in reference:")
        print(f"\nNumber of Atoms: {len(atoms_notinrefer)}\n")
        print(atoms_notinrefer)

        print("\nReses that have interactions however not included in reference:\n")

        res_notinrefer = list(set(res_notinrefer))
        print(f"\nNumber of Reses: {len(res_notinrefer)}\n")
        print(res_notinrefer)

        colname = []
        for iatm in self.reference_atom:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{iatm}_{iifp}')

        self.AAIFP_full = pd.DataFrame(
            _AAIFP_full, columns=colname, index=self.molecule_list)
        colname = []
        for ires in self.reference_res:
            for iifp in ['hbd', 'halg', 'elec', 'hrdr', 'pipi']:
                colname.append(f'{ires}_{iifp}')
        self.RESIFP_full = pd.DataFrame(
            _RESIFP_full, columns=colname, index=self.molecule_list)

        return self.AAIFP_full, self.RESIFP_full

    def compress_IFP(self, num):
        '''
        Compress IFP to remove not important descriptors 
        '''
        count_col = self.AAIFP_full.astype('int').sum()
        count_col.sort_values(ascending=False, inplace=True)
        pd_count_col = pd.DataFrame(count_col, columns=['counts'])
        pd_count_col.to_csv('Interaction_occurrence_count.csv')
        top_col = count_col[range(num)]
        top_AAIFP = self.AAIFP_full[top_col.index]
        return top_AAIFP

    def remove_allzeros(self):
        '''
        Remove the columns of all zeros 
        '''
        count_col = self.AAIFP_full.astype('int').sum()
        count_col.sort_values(ascending=False, inplace=True)
        pd_count_col = pd.DataFrame(count_col, columns=['counts'])
        pd_count_col = pd_count_col[pd_count_col['counts'] > 0]

        nonzero_AAIFP = self.AAIFP_full[pd_count_col.index]
        return nonzero_AAIFP
