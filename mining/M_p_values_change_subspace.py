import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from E_synthetic_data import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, \
    get_abcd_hyperparameters_from_str, cm2inch, move_legend_below_graph

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{helvet}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
# mpl.rc('font', family='serif')


if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    result_df = []
    j = 0
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    if os.path.exists(os.path.join(cache_dir, "cached-p-values.csv")):
        print("Use cache")
        result_df = pd.read_csv(os.path.join(cache_dir, "cached-p-values.csv"))
    else:
        for file in tqdm(all_files):
            j += 1
            # if j > 10:
            #     break
            df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",", index_col=False).convert_dtypes()
            df = fill_df(df)
            approach = np.unique(df["approach"])[0]
            if not "ABCD" in approach:
                continue
            params = np.unique(df["parameters"])[0]
            dataset = np.unique(df["dataset"])[0]
            dims = np.unique(df["ndims"])[0]
            df = df[df["change-point"]]
            if "ABCD" in approach:
                parsed_params = get_abcd_hyperparameters_from_str(params)
                delta, E, eta = parsed_params[0], parsed_params[1], parsed_params[2]
            else:
                E = np.nan
                eta = np.nan
            for rep, rep_data in df.groupby("rep"):
                for _, row in rep_data.iterrows():
                    if pd.isna(row["dims-p"]) or pd.isna(row["dims-gt"]):
                        continue
                    gt = str_to_arr(row["dims-gt"], dtype=int)
                    if len(gt) == 0:
                        print("No change subspace in ground truth of dataset {}!".format(dataset))
                    p = str_to_arr(row["dims-p"], dtype=float)
                    for i in range(dims):
                        result_df.append([dataset, delta, E, eta, rep, dims, i, params, approach, i in gt, p[i]])
        result_df = pd.DataFrame(data=result_df, columns=["dataset", "delta", "E", "eta", "rep", "dims", "dim", "params",
                                                          "approach", "has changed", "p-value"])
        result_df.to_csv(os.path.join(cache_dir, "cached-p-values.csv"), index=False)

    steps = 10
    max_thresh = 4
    evaluated_thresholds = max_thresh - max_thresh * np.arange(steps + 1) / steps
    result = []
    groupby = ["dataset", "approach", "params", "rep"]
    result_df = result_df[result_df["dataset"] != "RBF"]
    for keys, gdf in result_df.groupby(groupby):
        gt = gdf["has changed"]
        p = gdf["p-value"]
        for thresh in evaluated_thresholds:
            assert p.shape == gt.shape
            subspaces = p < thresh
            found_indices = subspaces[subspaces].index
            gt_indices = gt[gt].index
            intersect = np.intersect1d(gt_indices, found_indices)
            union = np.union1d(found_indices, gt_indices)
            jaccard = len(intersect) / len(union)
            prec = 0
            rec = 0
            f1 = 0
            tp = np.sum(np.logical_and(subspaces, gt))
            fp = np.sum(np.logical_and(subspaces, np.invert(gt)))
            fn = np.sum(np.logical_and(np.invert(subspaces), gt))
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            result.append(list(keys) + [jaccard, prec, rec, f1, thresh])
    result = pd.DataFrame(result, columns=groupby + ["Jaccard", "Prec.", "Rec.", "F1", r"$\delta_j$"])
    avg_df = result.groupby(r"$\delta_j$").mean().reset_index()
    avg_df["dataset"] = "Avg."
    result = pd.concat([result, avg_df]).reset_index()
    result = result.sort_values(by=["dataset"], ascending=False)
    n_colors = len(np.unique(result["dataset"]))
    sns.set_palette(sns.cubehelix_palette())
    result = result.melt(id_vars=["dataset", "approach", "params", "rep", r"$\delta_j$"],
                         value_vars=["Jaccard", "Prec.", "Rec.", "F1"],
                         var_name="Metric", value_name="Score")
    result = result[result["Metric"] != "Jaccard"].sort_values(by="Metric").sort_values(by="dataset")
    g = sns.relplot(data=result, x=r"$\delta_j$", y="Score", hue="dataset", col="Metric", kind="line", legend=False)
    plt.gcf().set_size_inches((3.5, 2))
    lines = g.figure.axes[-1].get_lines()
    labels = np.sort(np.unique(result["dataset"]))[::-1]
    plt.figlegend(lines, labels, loc='lower center', frameon=False, ncol=3, title=None)
    for i, ax in enumerate(g.figure.axes):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        col_title = ax.get_title().split(" = ")[-1]
        ax.set_title(col_title)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.47)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "delta_j_sensitivity.pdf"))
    plt.show()

