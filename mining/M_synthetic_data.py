import os

import numpy as np
import pandas as pd
from changeds.metrics import true_positives, false_positives, false_negatives, recall, precision, fb_score, \
    mean_until_detection, jaccard
from scipy.stats import spearmanr
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from E_synthetic_data import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def filter_best(df, worst: bool, median: bool, add_mean: bool = True):
    df = df.sort_values(by=["F1 (Region)", "Sp. Corr."])
    median_indices = []
    min_indices = []
    max_indices = []
    indices = []
    for _, gdf in df.groupby(["Dataset", "Approach", "Dims"]):
        if len(gdf.dropna()) == 0:
            indices.append(gdf.index[0])
            continue
        max_index = gdf["F1"].idxmax()
        indices.append(max_index)
        if "ABCD2" in gdf["Approach"].to_numpy():
            max_indices.append(max_index)
            if median:
                med = gdf["F1"].median()
                median_index = (gdf["F1"] - med).abs().idxmin()
                if median_index == max_index and len(gdf) > 1:
                    median_index = (gdf["F1"] - med).abs().drop(max_index).idxmin()
                median_indices.append(median_index)
            if worst:
                min_index = gdf["F1"].idxmin()
                min_indices.append(min_index)
    if median:
        indices += median_indices
        df["Approach"].loc[median_indices] = "ABCD2 (med)"
    if worst:
        indices += min_indices
        df["Approach"].loc[min_indices] = "ABCD2 (min)"
    indices = np.unique(indices)
    df["Approach"].loc[max_indices] = "ABCD2 (max)"
    df = df.loc[indices]
    if add_mean:
        df = add_mean_column(df)
    return df.reset_index()


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
            prec = len(np.intersect1d(a, b)) / len(a)
            rec = len(np.intersect1d(a, b)) / len(b)
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
    changes = df["change-point"]
    idxs = [i for i, change in enumerate(changes) if change]
    severities = df["severity-gt"].iloc[idxs]
    detected_severities = df["severity"].iloc[idxs]
    x = severities.to_numpy()
    y = detected_severities.to_numpy()
    if np.all(pd.isna(x)):
        return np.nan
    na_indices = [i for i in range(len(x)) if pd.isna(x[i])]
    x = np.delete(x, na_indices)
    y = np.delete(y, na_indices)
    # for i, s in zip(severities.index, severities.to_list()):
    #     for j, d in zip(detected_severities.index, detected_severities.to_list()):
    #         if pd.isna(s):  # TODO: Find out why this even happens
    #             continue
    #         if j < i:
    #             continue
    #         # s = str_to_arr(s, dtype=float)[0]
    #         x.append(s)
    #         y.append(d)
    #         break
    corr, p = spearmanr(x, y)
    return corr


if __name__ == '__main__':
    print_summary = True
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    if os.path.exists(os.path.join(cache_dir, "cached.csv")):
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
            for rep, rep_data in df.groupby("rep"):
                true_cps = [i for i in rep_data.index if rep_data["is-change"].loc[i]]
                cp_distance = true_cps[0]
                n_seen_changes = len(true_cps)
                reported_cps = [i for i in rep_data.index if rep_data["change-point"].loc[i]]
                f1 = fb_score(true_cps=true_cps, reported_cps=reported_cps, T=2000)
                jac, region_prec, region_rec = compute_region_metrics(rep_data)
                f1_region = 2 * (region_prec * region_prec) / (region_rec + region_prec)
                severity = compute_severity_metric(df)
                result_df.append([
                    dataset, dims, approach, params, f1, f1_region, region_prec, region_rec, jac, severity
                ])
        result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "Parameters", "F1", "F1 (Region)", "Prec. (Region)",
                                                     "Rec. (Region)", "Jaccard", "Sp. Corr."])
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters", "Dims"]).mean().reset_index()
    result_df = filter_best(result_df, median=False, worst=False)
    result_df = result_df[result_df["Approach"] != "ABCD2"]  # we only report the result w.r.t. max F1 as we do for our competitors.
    sort_by = ["Dataset", "Dims", "Approach"]
    result_df = result_df.sort_values(by=sort_by)
    print(result_df.groupby(["Dataset", "Approach", "Dims"]).mean().round(
        decimals={"F1 (Region)": 2, "Prec. (Region)": 2, "Rec. (Region)": 2, "Jaccard": 2, "Sp. Corr.": 2}
    ).to_latex())
    result_df.fillna(0.0, inplace=True)
    sns.relplot(data=result_df.reset_index(), x="Sp. Corr.", y="F1 (Region)", hue="Dims", style="Approach", col="Dataset", kind="scatter")
    # plt.gcf().set_size_inches(6, 2.5)
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    for ax in plt.gcf().axes:
        ax.axhline(0, c="black", lw=0.5)
        ax.axvline(0, c="black", lw=0.5)
    plt.tight_layout()
    plt.show()
