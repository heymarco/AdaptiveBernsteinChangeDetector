import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from changeds.metrics import fb_score, jaccard, true_positives, false_positives, precision, recall, false_negatives
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score

from E_synthetic_data import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, \
    get_abcd_hyperparameters_from_str, cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def compute_region_metrics(df: pd.DataFrame):
    if not np.any(pd.isnull(df["dims-found"]) == False):
        return np.nan, np.nan, np.nan
    changes = df["change-point"]
    idxs = [i for i, change in enumerate(changes) if change]
    regions_gt = df["dims-gt"].iloc[idxs]
    regions_detected = df["dims-found"].iloc[idxs]
    jaccard_scores = []
    prec_scores = []
    recall_scores = []
    for a, b in zip(regions_gt, regions_detected):
        try:
            a = str_to_arr(a, int)
            b = str_to_arr(b, int)
            jac = jaccard(a, b) if len(b) > 0 else np.nan
            tp = len(np.intersect1d(a, b))
            fp = len(np.setdiff1d(b, np.intersect1d(a, b)))
            fn = len(np.setdiff1d(a, np.intersect1d(a, b)))
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            jaccard_scores.append(jac)
            prec_scores.append(prec)
            recall_scores.append(rec)
        except:
            continue
    jac = np.nanmean(jaccard_scores) if len(jaccard_scores) > 0 else np.nan
    prec = np.nanmean(prec_scores) if len(prec_scores) > 0 else np.nan
    rec = np.nanmean(recall_scores) if len(recall_scores) > 0 else np.nan
    return jac, prec, rec


def compute_severity_metric(df: pd.DataFrame):
    df = df[df["change-point"].astype(bool)]
    severities = df["severity-gt"]
    detected_severities = df["severity"]
    x = severities.to_numpy()
    y = detected_severities.to_numpy()
    if np.all(pd.isna(x)):
        return np.nan
    na_indices = [i for i in range(len(x)) if pd.isna(x[i])]
    x = np.delete(x, na_indices)
    y = np.delete(y, na_indices)
    rep = np.unique(df["rep"])
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        corr, p = spearmanr(x, y)
    except:
        x = np.array([float(x_i[1:-1]) for x_i in x.flatten()])
        y = y.flatten().astype(float)
        corr, p = spearmanr(x, y)
    return corr


if __name__ == '__main__':
    print_summary = True
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    if os.path.exists(os .path.join(cache_dir, "cached.csv")):
        print("Use cache")
        result_df = pd.read_csv(os.path.join(cache_dir, "cached.csv"))
    else:
        result_df = []
        j = 0
        for file in tqdm(all_files):
            j += 1
            # if j > 10:
            #     break
            df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",").convert_dtypes()
            df = fill_df(df)
            approach = np.unique(df["approach"])[0]
            params = np.unique(df["parameters"])[0]
            dataset = np.unique(df["dataset"])[0]
            dims = np.unique(df["ndims"])[0]
            severity = compute_severity_metric(df)
            if "ABCD" in approach or "ABCD0" in approach:
                parsed_params = get_abcd_hyperparameters_from_str(params)
                E, eta = parsed_params[1], parsed_params[2]
            else:
                E = np.nan
                eta = np.nan
            result_df.append([np.nan, dataset, dims, approach, params, E, eta,
                              np.nan, np.nan, np.nan, np.nan, severity, np.nan])
            for rep, rep_data in df.groupby("rep"):
                true_cps = [i for i in rep_data.index if rep_data["is-change"].loc[i]]  # TODO: check if this is correct
                cp_distance = 2000
                if "MNIST" in dataset or "CIFAR" in dataset:
                    cp_distance = 4000
                reported_cps = [i for i in rep_data.index if rep_data["change-point"].loc[i]]
                tp = true_positives(true_cps, reported_cps, cp_distance)
                fp = false_positives(true_cps, reported_cps, cp_distance)
                fn = false_negatives(true_cps, reported_cps, cp_distance)
                n_seen_changes = len(true_cps)
                delays = rep_data["delay"].loc[reported_cps].tolist()
                prec = precision(tp, fp, fn)
                rec = recall(tp, fp, fn)
                f1 = fb_score(true_cps, reported_cps, T=cp_distance)
                jac, region_prec, region_rec = compute_region_metrics(rep_data)
                f1_region = 2 * (region_rec * region_prec) / (region_rec + region_prec)
                result_df.append([
                    rep, dataset, dims, approach, params, E, eta, f1_region, region_prec, region_rec, jac, np.nan, f1
                ])
        result_df = pd.DataFrame(result_df, columns=["Rep", "Dataset", "Dims", "Approach", "Parameters", "E", "eta", "F1 (Subspace)", "Prec. (Region)",
                                                     "Rec. (Region)", "Jaccard", "Pearson R", "F1"])
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    # result_df.groupby(["Dataset", "Approach", "Dims"])["Pearson R"].fillna(method="ffill", inplace=True)
    result_df = result_df[result_df["Dataset"] != "RBF"]
    for _, gdf in result_df.groupby(["Approach", "Dataset", "Dims"]):
        index = gdf.index
        gdf = gdf["Pearson R"].fillna(method="ffill")
        result_df["Pearson R"].loc[index] = gdf
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters", "Dims"]).mean().reset_index()
    # med_dfs = []
    # for _, gdf in result_df.groupby(["Dataset", "Approach", "Dims"]):
    #     med_f1 = gdf["F1"].median()
    #     gdf["Pearson R"][gdf["F1"] < med_f1] = np.nan
    #     gdf["Jaccard"][gdf["F1"] < med_f1] = np.nan
    #     med_dfs.append(gdf)
    # result_df = pd.concat(med_dfs)
    sort_by = ["Dataset", "Dims", "Approach"]
    result_df = result_df.sort_values(by=sort_by)
    print(result_df.groupby(["Dataset", "Approach", "Dims"]).mean().round(
        decimals={"F1 (Subspace)": 2, "Prec. (Region)": 2, "Rec. (Region)": 2, "Jaccard": 2, "Pearson R": 2}
    ))
    result_df = result_df.sort_values(by=["Dims", "Dataset"])
    result_df = result_df.astype(dtype={"Dims": str})
    result_df["Pearson R"][result_df["Approach"] == "D3"] = result_df[result_df["Approach"] == "D3"]["Pearson R"]
    result_df = result_df[result_df["Dataset"] != "Average"]
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD"
    result_df = result_df.sort_values(by="Approach")
    n_colors = len(np.unique(result_df["Approach"]))
    palette = sns.cubehelix_palette(n_colors=n_colors)
    melted_df = pd.melt(result_df, id_vars=["Dataset", "Approach"], value_vars=["F1", "Jaccard", "Pearson R"],
                        var_name="Metric", value_name="Value")
    melted_df = melted_df[melted_df["Value"].isna() == False]
    g = sns.catplot(data=melted_df, x="Approach", y="Value", col="Dataset", row="Metric", kind="box",
                    linewidth=0.7, fliersize=2, sharey="row", showfliers=False, palette=palette)
    g.set(xlabel=None)
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i == 0:
            ax.set_ylabel("F1")
        if i == 4:
            ax.set_ylabel("Jaccard")
        if i == 8:
            ax.set_ylabel("Pearson R")
        if i < 4:
            col_title = ax.get_title().split(" = ")[-1]
            if col_title.startswith("Normal"):
                a, b = col_title.split("al")
                col_title = a + "." + b
            ax.set_title(col_title)
        else:
            ax.set_title("")
    plt.gcf().set_size_inches(3.5, 3.5 * 1.2)
    plt.tight_layout()
    plt.subplots_adjust(left=0.16, wspace=0.15, right=0.99)
    plt.savefig(os.path.join("..", "figures", "evaluation_drift_region.pdf"))
    plt.show()
