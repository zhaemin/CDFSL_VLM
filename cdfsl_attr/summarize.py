desc = """
This scripts summarizes the experiments under a specified root folder (./results by default.)
You can customize it by only computing results for:
1. a specific setting with the --setting argument (base2new or standard);
2. an experiment of your choice (--exp_name, which should match those passed to main.py)

The script will NOT automatically determine if a (dataset, backbone, seed) combination is missing or incomplete. 
You are responsible to do so :)

A summary {setting}_{exp_name}.csv is dumped to the root of this repository at end.
"""


import pandas as pd
from pathlib import Path
from argparse import ArgumentParser


def parse_path(path: Path):
    df = pd.read_csv(path)
    return df


def hmean(a, b):
    return 2*a*b / (a+b)


def main(args):

    # load all result files under the root that meet the required experiment name
    root = Path(args.root)
    result_files = []
    for leaf in root.rglob(f"{args.exp_name}.csv"):
        result_files.append(parse_path(leaf))
    df = pd.concat(result_files, ignore_index=True)

    # filter according to the setting
    print(f"{len(df)} total files loaded.")
    df = df[df['setting'] == args.setting]
    print(f"{len(df)} files remained after filtering for setting {args.setting}.")

    # print some stats here, maybe helps users in spotting missing combos
    modes = df['mode'].unique()
    backbones = df['backbone'].unique()
    datasets = df['dataset'].unique()
    shots = df['shots'].unique()
    print(
        "Detected:",
        f"{len(modes)} unique modes (methods);",
        f"{len(backbones)} unique backbones;",
        f"{len(datasets)} unique datasets;",
        f"{len(shots)} unique shot availabilities",
        sep="\n - ", end="\n\n"
    )

    # now we can group according to the backbone and the mode
    grouped = df.groupby(['mode', 'backbone', 'dataset', 'shots', 'exp_name'])

    # reduce results across different seeds and report the average
    if args.setting == 'standard':
        agg = grouped.agg({
            'acc_test': ['mean', 'std'],
            'runtime': ['mean', 'std']
        }).reset_index()
        agg.columns = ['_'.join(col) if all(col) else col[0] for col in agg.columns]
    elif args.setting == 'base2new':
        agg = grouped.agg({
            'acc_test_base': ['mean', 'std'],
            'acc_test_new': ['mean', 'std'],
            'runtime': ['mean', 'std']
        }).reset_index()    
        agg.columns = ['_'.join(col) if all(col) else col[0] for col in agg.columns]
        agg['hmean'] = hmean(agg['acc_test_base_mean'], agg['acc_test_new_mean'])

    # within each group of (dataset, setting, shots), sort s.t. you see the best method first
    # (regardless of the exp_name, maybe you wanna compare different exps with different settings 
    # for the same method)
    metric = "hmean" if args.setting == "base2new" else "acc_test_mean"
    agg = agg.sort_values(by=['dataset', 'backbone', 'shots', metric], ascending=False)
    
    # dump the summary on disk, divided per backbone :)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for backbone in agg['backbone'].unique():
        sliced = agg[agg['backbone'] == backbone]
        outpath = outdir / f"{args.setting}_{args.exp_name}_{backbone.replace('/', '-')}.csv"
        sliced.round(2).to_csv(outpath, index=False)
        print(f"Results saved at {outpath}")


if __name__ == "__main__":
    parser = ArgumentParser(description=desc)
    parser.add_argument("--setting", type=str, choices=["base2new", "standard"], default="standard")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--root", type=str, default="./results")
    parser.add_argument("--outdir", type=str, default="./summaries")

    args = parser.parse_args()
    main(args)