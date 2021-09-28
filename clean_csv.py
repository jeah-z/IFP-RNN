import pandas as pd
import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", help="csv file to be processed",
                        default='')
    parser.add_argument(
        "--dupl_cols", help="duplicated columns, exp. 'ligands','smi'")
    parser.add_argument(
        "--del_col", help="if the columns euqual the value the line will be deleted",   type=str, default='')
    parser.add_argument(
        "--del_vals", help="abosolute path of the target pdb for aligning",   type=str, default='')
    parser.add_argument(
        "--save_csv", help="the results csv file",   type=str, default='')
    parser.add_argument(
        "--smi_size", help="the smallest size of the smiles.", type=int, default=15)
    args = parser.parse_args()
    return args


def main(args):
    df = pd.read_csv(args.csv_file)
    if args.dupl_cols != '':
        print(f"Deleting the duplicates rows!")
        print(f"The input csv size is : {len(df)} rows!")
        dupl_cols = args.dupl_cols.strip().split(",")
        df.drop_duplicates(dupl_cols, "first", inplace=True)
        print(f"After dropping duplicates, the csv size is : {len(df)} rows!")
    if args.del_col != '' and args.del_vals != '':
        del_col = args.del_col
        del_vals = args.del_vals.strip().split(",")

        print(f"Values of {del_col} will be removed: {del_vals}!")
        print(df[del_col])
        df = df[~df[del_col].isin(del_vals)]
        small_idx = []
        for idx, row in df.iterrows():
            if len(row[del_col]) < args.smi_size:
                small_idx.append(idx)
        df.drop(small_idx, inplace=True)
        df_group = pd.DataFrame(df.groupby([del_col]).count())
        df_group = pd.DataFrame(df_group.sort_values(
            dupl_cols[0], ascending=False))
        df_group = df_group[:20]
        print(df_group)

        print(
            f"After deleting according value, the csv size is : {len(df)} rows!")
    df.to_csv(args.save_csv, index=None)


if __name__ == '__main__':
    args = get_parser()
    main(args)
