from __future__ import print_function

import sys
import os
import argparse
try:
    # Open Babel >= 3.0
    from openbabel import openbabel as ob
except ImportError:
    import openbabel as ob
# import pybel
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def write_sdf(smi_csv, sdf_file):
    df = pd.read_csv(smi_csv)
    writer = Chem.SDWriter(sdf_file)
    for idx, row in df.iterrows():
        print(f"preparing the SMILES: {row['SMILES']}")
        mol = Chem.MolFromSmiles(row["SMILES"])
        print(f"setting the name: {row['Name']}")
        mol.SetProp('_Name', row['Name'])
        AllChem.Compute2DCoords(mol)
        writer.write(mol)
    writer.close()


def main(args):
    write_sdf(args.smi_csv, args.sdf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smi_csv", help="csv file of SMILES", default='')
    parser.add_argument(
        "--sdf", help="SDF file to save the results", default='./')
    args = parser.parse_args()
    main(args)
