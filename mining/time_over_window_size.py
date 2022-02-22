import os
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

sys.path.append(os.getcwd())
from util import get_last_experiment_dir, move_legend_below_graph


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def filter_best(df, worst: bool, median: bool):
    median_indices = []
    min_indices = []
    max_indices = []
    indices = []
    for _, gdf in df.groupby(["Dataset", "Approach"]):
        indices.append(gdf["F1"].idxmax())
        if "ABCD2" in gdf["Approach"].to_numpy():
            if median:
                med = gdf["F1"].median()
                median_index = (gdf["F1"] - med).abs().idxmin()
                median_indices.append(median_index)
            if worst:
                min_index = gdf["F1"].idxmin()
                min_indices.append(min_index)
            max_index = gdf["F1"].idxmax()
            max_indices.append(max_index)
    if median:
        indices += median_indices
        df["Approach"].loc[median_indices] = "ABCD2 (med)"
    if worst:
        indices += min_indices
        df["Approach"].loc[min_indices] = "ABCD2 (min)"
    indices = np.unique(indices)
    df["Approach"].loc[max_indices] = "ABCD2 (max)"
    df = df.loc[indices]
    return df


if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir()
    all_files = os.listdir(last_exp_dir)
    result_df = []
    ws_str = r"$|\mathcal{W}|$"
    j = 0
    for file in tqdm(all_files):
        df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",").convert_dtypes().ffill()
        approach = np.unique(df["approach"])[0]
        j += 1
        # if j > 130:
        #     break
        if approach != "ABCD2":
            continue
        params = np.unique(df["parameters"])[0]
        dataset = np.unique(df["dataset"])[0]
        dims = np.unique(df["ndims"])[0]
        for rep, rep_data in df.groupby("rep"):
            true_cps = [i for i in range(len(rep_data)) if rep_data["is-change"].iloc[i]]
            cp_distance = true_cps[0]
            reported_cps = [i for i in range(len(rep_data)) if rep_data["change-point"].iloc[i]]
            diff = round(rep_data["time"].diff() / 10e6, 3)
            ws = rep_data["w1"] + rep_data["w2"]
            [result_df.append([dataset, dims, approach, d, w]) for d, w in zip(diff, ws)]
    result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "MTPO [ms]", ws_str])
    outlier_thresh = np.percentile(result_df["MTPO [ms]"].dropna(), 95)
    result_df = result_df[result_df["MTPO [ms]"].fillna(np.infty) < outlier_thresh]
    all_datasets = np.unique(result_df["Dataset"])
    result_df.dropna(inplace=True)
    result_df = result_df[result_df[ws_str] < 8000]
    result_df.sort_values(by=[ws_str])
    result_df["bins"] = (result_df[ws_str] / 20).astype(int)
    result_df = result_df.groupby(["Dataset", "Approach", "bins", "Dims"]).mean()
    result_df.sort_values(by="Dims", inplace=True)
    sns.lineplot(data=result_df, x=ws_str, y="MTPO [ms]", hue="Dataset",
                 ci=None, palette=sns.cubehelix_palette(n_colors=len(all_datasets)))
    plt.gcf().set_size_inches((4, 2))
    move_legend_below_graph(np.array([plt.gca()]), ncol=3, title="")
    plt.tight_layout()
    # plt.savefig(os.path.join(os.getcwd(), "..", "figures", "mtpo_ws.pdf"))
    plt.show()
