import pandas as pd
import argparse



def main(args):
    csv_files=args.input.split(',')
    for csv_file in csv_files:
        print(f'Processing {csv_file}')
        df=pd.read_csv(csv_file)
        df=df.set_index('Molecule')
        idx_select=[]
        for idx,row in df.iterrows():
            idx_sp=idx.split('_')
            if len(idx_sp)<2:
                idx_select.append(idx)
            elif int(idx_sp[-1])<args.max_idx:
                idx_select.append(idx)
        new_df=df.loc[idx_select]
        new_df.to_csv(csv_file.replace('.csv',f'_{args.max_idx}pose.csv'))



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="The IFP file with poses that needed to be removed", type=str, default='')
    parser.add_argument("--max_idx", type=int,
                        help="maximum pose index", default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # test section

    args = get_parser()
    main(args)