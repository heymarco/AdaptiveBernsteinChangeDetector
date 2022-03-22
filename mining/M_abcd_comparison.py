import os

import numpy as np
import pandas as pd
from changeds.metrics import true_positives, false_positives, false_negatives, recall, precision, fb_score, \
    mean_until_detection, jaccard, mean_cp_detection_time_error
from scipy.stats import spearmanr
from tqdm import tqdm

from E_sensitivity_study import ename
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
    for _, gdf in df.groupby(["Dataset", "Approach"]):
        if len(gdf.dropna()) == 0:
            indices.append(gdf.index[0])
            continue
        max_index = gdf["F1 (Region)"].idxmax()
        indices.append(max_index)
        if "ABCD2" in gdf["Approach"].to_numpy():
            max_indices.append(max_index)
            if median:
                med = gdf["F1 (Region)"].median()
                median_index = (gdf["F1 (Region)"] - med).abs().idxmin()
                if median_index == max_index and len(gdf) > 1:
                    median_index = (gdf["F1 (Region)"] - med).abs().drop(max_index).idxmin()
                median_indices.append(median_index)
            if worst:
                min_index = gdf["F1 (Region)"].idxmin()
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
    x = []
    y = []
    for i, s in zip(severities.index, severities.to_list()):
        for j, d in zip(detected_severities.index, detected_severities.to_list()):
            if pd.isna(s):  # TODO: Find out why this even happens
                continue
            if j < i:
                continue
            s = str_to_arr(s, dtype=float)[0]
            x.append(s)
            y.append(d)
            break
    corr, p = spearmanr(x, y)
    return corr


def mean_time_per_example(df):
    runtime = df["time"].iloc[-1] - df["time"].iloc[0]
    observations = df.index[-1] - df.index[0]
    return runtime / observations


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
            df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",", index_col=0).convert_dtypes()
            df = fill_df(df)
            approach = np.unique(df["approach"])[0]
            params = np.unique(df["parameters"])[0]
            dataset = np.unique(df["dataset"])[0]
            dims = np.unique(df["ndims"])[0]
            for rep, rep_data in df.groupby("rep"):
                true_cps = [i for i in rep_data.index if rep_data["is-change"].loc[i]]  # TODO: check if this is correct
                cp_distance = true_cps[1] - true_cps[0]
                reported_cps = [i for i in rep_data.index if rep_data["change-point"].loc[i]]
                tp = true_positives(true_cps, reported_cps, cp_distance)
                fp = false_positives(true_cps, reported_cps, cp_distance)
                fn = false_negatives(true_cps, reported_cps, cp_distance)
                n_seen_changes = len(true_cps)
                jac, region_prec, region_rec = compute_region_metrics(rep_data)
                f1_region = 2 * (region_prec * region_prec) / (region_rec + region_prec)
                severity = compute_severity_metric(rep_data)
                delays = rep_data["delay"].loc[reported_cps].tolist()
                prec = precision(tp, fp, fn)
                rec = recall(tp, fp, fn)
                f1 = fb_score(true_cps, reported_cps, T=2000)
                mttd = mean_until_detection(true_cps, reported_cps)
                mae_delay = mean_cp_detection_time_error(true_cps, reported_cps, delays)
                mtpe = mean_time_per_example(rep_data)
                mtpe = mtpe / 10e6
                result_df.append([
                    dataset, dims, approach, params,
                    f1, mttd, mae_delay,
                    f1_region,
                    severity,
                    mtpe
                ])
        result_df = pd.DataFrame(result_df, columns=[
            "Dataset", r"$d$", "Approach", "Parameters",
            "F1", "MTTD", r"MAE $\tau$",
            "F1 (Region)",
            "Sp. Corr.",
            "MTPO [ms]"])
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    sort_by = ["Dataset", r"$d$", "Approach", "Parameters"]
    result_df = result_df.sort_values(by=sort_by)
    result_df.drop(["Parameters"], axis=1, inplace=True)
    print(result_df.groupby(["Approach", "Dataset", r"$d$"]).mean().round(
        decimals=2
    ).to_latex(escape=False))
