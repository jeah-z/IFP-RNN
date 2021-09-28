try:
    from openbabel import pybel
except:
    import pybel
import pandas as pd
import argparse
from functools import partial
from multiprocessing import Pool
from tqdm.auto import tqdm
import os
from pathlib import Path
import rdkit
from rdkit import Chem
from pebble import concurrent, ProcessPool
from concurrent.futures import TimeoutError
import glob

glide = "/mnt/home/zhangjie/Bin/Schrodinger2017/glide"
structconvert = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/structconvert"
prepwizard = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/prepwizard"
glide_sort = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/glide_sort"
mol2convert = "/mnt/home/zhangjie/Bin/Schrodinger2017/utilities/mol2convert"
prepare_ligand4 = '/mnt/home/zhangjie/Bin/MGLTools-1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py'


def find_files(suffix):
    '''find all the files with specific suffix.
    '''
    files = os.listdir('./')
    file_list = []
    for file in files:
        file_sp = file.split('_')
        if len(file_sp) < 2:
            print(file+'\t was omitted!')
            continue
        elif file_sp[-1] == suffix:
            file_list.append(file)
    print(f"{len(file_list)} files have been detected!")
    return file_list


def combine_pdbqtReport(pdbqt_file, report_file):
    '''This script fetch docking score from the report file and write the dockind score to the specific pose of the pdbqt file.
    '''
    with open(report_file, 'r') as report_file_f:
        parse_mark = 0
        dockScore_list = []
        for line in report_file_f.readlines():
            line_sp = line.strip().split()
            if parse_mark > 0 and len(line_sp) == 0:
                break
            if len(line_sp) > 0:
                if line_sp[0] == '====':
                    parse_mark += 1
                    continue
                if len(line_sp) == 19 and parse_mark > 0:
                    dockScore_list.append([line_sp[0], line_sp[3], line_sp[1]])

    pdbqt_newFile = pdbqt_file.replace(".pdbqt", "_out.pdbqt")
    pdbqt_newFile_f = open(pdbqt_newFile, 'w')
    with open(pdbqt_file, 'r') as pdbqt_file_f:
        if len(dockScore_list) == 1:
            glide_score = dockScore_list.pop(0)
            pdbqt_newFile_f.write(
                f'REMARK VINA RESULT:      {glide_score[1]}      0.000      0.000\n')
            for line in pdbqt_file_f.readlines():
                pdbqt_newFile_f.write(line)
        else:
            for line in pdbqt_file_f.readlines():
                line_sp = line.strip().split()
                if len(line_sp) > 0:
                    if line_sp[0] == 'MODEL' and len(line_sp) == 2:
                        pdbqt_newFile_f.write(line)
                        glide_score = dockScore_list.pop(0)
                        assert int(glide_score[0]) == int(line_sp[1])
                        pdbqt_newFile_f.write(
                            f'REMARK VINA RESULT:      {glide_score[1]}      0.000      0.000\n')
                    else:
                        pdbqt_newFile_f.write(line)
        assert len(
            dockScore_list) == 0  # '''This is to make sure no glide score was left! '''
    pdbqt_newFile_f.close()


def mae_toPdbqt(imaegz, pdbqt_dir):
    isimple_name = imaegz.replace('_pv.maegz', '')
    if os.path.exists(f'../{pdbqt_dir}/{isimple_name}_out.pdbqt'):
        print(f"{imaegz} have been processed before!")
        return 0
    # if maegz_files.index(imaegz) > 10:  # This is to accelerate the test speed!
    #     break

    os.system(
        f'{glide_sort} -r ../{pdbqt_dir}/{isimple_name}.rept {imaegz} -o ../{pdbqt_dir}/{isimple_name}.mae')
    os.system(
        f'{mol2convert} -n 2: -imae ../{pdbqt_dir}/{isimple_name}.mae -omol2 ../{pdbqt_dir}/{isimple_name}.mol2')
    os.system(
        f'babel -imol2 ../{pdbqt_dir}/{isimple_name}.mol2 -opdbqt ../{pdbqt_dir}/{isimple_name}.pdbqt')
    combine_pdbqtReport(
        f'../{pdbqt_dir}/{isimple_name}.pdbqt', f'../{pdbqt_dir}/{isimple_name}.rept')
    # clean the unnecessary files
    for isufix in ['rept', 'mol2', 'mae', 'pdbqt']:
        os.system(
            f"rm ../{pdbqt_dir}/{isimple_name}.{isufix}")


def main(args):
    os.chdir(f"{args.path}")
    maegz_files = find_files('pv.maegz')
    pdbqt_dir = args.path.split("/")[-1]+'_pdbqt'
    os.system(f"mkdir ../{pdbqt_dir}")

    with ProcessPool(max_workers=args.n_jobs) as pool:
        print("RUNING POOL!!!!")
        for imaegz in maegz_files:
            future = pool.schedule(
                mae_toPdbqt, args=[imaegz, pdbqt_dir], timeout=600)
    '''Further clean the pdbqt folder, some files cannot be removed for a unkown reason!'''
    os.chdir(f"../{pdbqt_dir}")
    files = os.listdir('./')
    for ifile in files:
        ifile_sp = ifile.split(".")
        if "_" not in ifile and ifile_sp[-1] in ['rept', 'mol2', 'mae', 'pdbqt']:
            os.system(f'rm {ifile}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="The directory of the docking results (maegz format).", type=str,
                        default='')
    parser.add_argument("--n_jobs", type=int,
                        help="cpu cores", default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # test section

    args = get_parser()
    main(args)
