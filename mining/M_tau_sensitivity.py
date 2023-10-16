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

sns.set_theme(context="paper", style="ticks", palette="deep")

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    result_df = []
    j = 0
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    p_values_exist = os.path.exists(os.path.join(cache_dir, "cached-p-values.csv"))
    tau_exists = os.path.exists(os.path.join(cache_dir, "cached_tau_results.csv"))
    if p_values_exist and not tau_exists:
        print("Use cache")
        result_df = pd.read_csv(os.path.join(cache_dir, "cached-p-values.csv"))
    if not p_values_exist:
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
        result = []
        groupby = ["dataset", "approach", "params", "rep"]
        result_df = result_df[result_df["dataset"] != "RBF"]

    if tau_exists:
        result = pd.read_csv(os.path.join(cache_dir, "cached_tau_results.csv"))
    else:
        steps = 40
        max_thresh = 4
        evaluated_thresholds = np.arange(0, max_thresh * steps + 1) / steps
        print(evaluated_thresholds)
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
                tp = np.sum(np.logical_and(subspaces, gt))
                fp = np.sum(np.logical_and(subspaces, np.invert(gt)))
                fn = np.sum(np.logical_and(np.invert(subspaces), gt))
                tn = np.sum(np.logical_and(np.invert(gt), np.invert(subspaces)))
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1 = 2 * prec * rec / (prec + rec)
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                result.append(list(keys) + [accuracy, jaccard, prec, rec, f1, thresh])
        result = pd.DataFrame(result, columns=groupby + ["Accuracy", "Jaccard", "Prec.", "Rec.", "F1", r"$\tau$"])
        result.to_csv(os.path.join(cache_dir, "cached_tau_results.csv"), index=False)

    avg_df = result.copy().groupby([r"$\tau$", "approach"]).mean().reset_index()
    avg_df["dataset"] = "Avg."
    result = pd.concat([result, avg_df]).reset_index()
    result = result.sort_values(by=["dataset"], ascending=False)
    n_colors = len(np.unique(result["dataset"]))
    # result = result.melt(id_vars=["dataset", "approach", "params", "rep", r"$\tau$"],
    #                      value_vars=["Jaccard", "Prec.", "Rec.", "F1"],
    #                      var_name="Metric", value_name="Score")
    # result = result[result["Metric"] != "Jaccard"].sort_values(by="Metric").sort_values(by="dataset", ascending=False)
    # result[result["Metric"] == "F1"] = "F1 (subspace)"
    result = result.sort_values(by="approach").sort_values(by="dataset", ascending=False)
    g = sns.relplot(data=result, x=r"$\tau$", y="Accuracy", hue="dataset",
                    col="approach",
                    height=cm2inch(3.5)[0], aspect=1.4,
                    kind="line", legend=False, ci=False)
    # plt.gcf().set_size_inches(cm2inch((16, 4)))
    # max_vals = result.groupby(["approach", r"$\tau$"]).mean().max().reset_index
    for i, ax in enumerate(g.figure.axes):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        col_title = ax.get_title().split(" = ")[-1]
        col_title = col_title.split("0")[0] + col_title.split("0")[1]
        col_title = col_title.split("(")[-1][:-1].upper()
        if col_title == "KPCA":
            col_title = "Kernel-PCA"
        ax.set_title(col_title)
        # max_tau = max_vals[max_vals["Approach"] == col_title].iloc[0][r"$\tau$"]
        ax.axhline(0.25, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axhline(0.75, color="gray", lw=0.7, ls="--", zorder=0)
        # plt.axvline(max_tau, lw=0.7, zorder=0)
        ax.set_xticks([0, 2, 4])
        ax.set_xticklabels([0, 2, 4])
    lines = g.figure.axes[-1].get_lines()
    labels = np.unique(result["dataset"]).tolist()
    labels = pd.Series(labels).sort_values(ascending=False)
    # titles = ["F1", "Rec.", "Prec."]
    plt.figlegend(lines, labels, loc='center right', frameon=False, ncol=1, title=None)
    plt.tight_layout(pad=.5)
    plt.gcf().subplots_adjust(right=.77)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "tau_sensitivity.pdf"))
    plt.show()

