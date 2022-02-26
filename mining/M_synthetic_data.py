import os

import numpy as np
import pandas as pd
from changeds.metrics import true_positives, false_positives, false_negatives, recall, precision, fb_score, \
    mean_until_detection, jaccard
from scipy.stats import spearmanr
from tqdm import tqdm

from E_synthetic_data import ename
from util import get_last_experiment_dir, str_to_arr


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def filter_best(df, worst: bool, median: bool, add_mean: bool = True):
    df = df.sort_values(by=["Jaccard", "MTTD"])
    median_indices = []
    min_indices = []
    max_indices = []
    indices = []
    for _, gdf in df.groupby(["Dataset", "Approach"]):
        max_index = gdf["Jaccard"].idxmax()
        indices.append(max_index)
        if "ABCD2" in gdf["Approach"].to_numpy():
            max_indices.append(max_index)
            if median:
                med = gdf["Jaccard"].median()
                median_index = (gdf["Jaccard"] - med).abs().idxmin()
                if median_index == max_index:
                    median_index = (gdf["Jaccard"] - med).abs().drop(max_index).idxmin()
                median_indices.append(median_index)
            if worst:
                min_index = gdf["Jaccard"].idxmin()
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
        return np.nan
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
    severities = df["severity-gt"].loc[idxs]
    detected_severities = df[not df["severity"].isnull()]["severity"]
    results = []
    for i, s in enumerate(severities):
        for j, d in enumerate(detected_severities):
            if d.index < s.index:
                continue
            s = str_to_arr(s, dtype=float)
            d = str_to_arr(d, dtype=float)
            corr, p = spearmanr(s, d)
            results.append(corr)
    return np.nanmean(results)


if __name__ == '__main__':
    print_summary = True
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    result_df = []
    j = 0
    for file in tqdm(all_files):
        j += 1
        # if j > 10:
        #     break
        df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",").convert_dtypes().ffill()
        approach = np.unique(df["approach"])[0]
        params = np.unique(df["parameters"])[0]
        dataset = np.unique(df["dataset"])[0]
        dims = np.unique(df["ndims"])[0]
        for rep, rep_data in df.groupby("rep"):
            true_cps = [i for i in range(len(rep_data)) if rep_data["is-change"].iloc[i]]
            cp_distance = true_cps[0]
            n_seen_changes = len(true_cps)
            reported_cps = [i for i in range(len(rep_data)) if rep_data["change-point"].iloc[i]]
            tp = true_positives(true_cps, reported_cps, cp_distance)
            fp = false_positives(true_cps, reported_cps, cp_distance)
            fn = false_negatives(true_cps, reported_cps, cp_distance)
            prec = precision(tp, fp, fn)
            rec = recall(tp, fp, fn)
            f1 = fb_score(true_cps, reported_cps, T=2000)
            mttd = mean_until_detection(true_cps, reported_cps)
            jac, region_prec, region_rec = compute_region_metrics(rep_data)
            severity = compute_severity_metric(df)
            result_df.append([
                dataset, dims, approach, params, f1, prec, rec,
                jac, region_prec, region_rec, severity, n_seen_changes
            ])
    result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "Parameters", "F1", "Prec.",
                                                 "Rec.", "RCD", "Jaccard", "MTTD", "MTPO [ms]", "PC"])
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters"]).mean().reset_index()
    if print_summary:
        summary = filter_best(result_df, median=True, worst=False)
    result_df = result_df.round(decimals={
        "F1": 2, "Prec.": 2, "Rec.": 2, "RCD": 2, "Jaccard": 2, "MTPO [ms]": 3, "MTTD": 1
    })
    result_df[result_df["Dataset"] == "Average"] = 0
    sort_by = ["Dataset", "Dims", "Approach", "Parameters"]
    result_df = result_df.sort_values(by=sort_by)
    result_df.drop(["Jaccard", "Parameters", "Dims"], axis=1, inplace=True)
    result_df["PC"] = result_df["PC"].astype(int)
    print(result_df.set_index(["Dataset", "Approach"]).to_latex(escape=False))
    if print_summary:
        summary = summary.round(decimals={
            "F1": 2, "Prec.": 2, "Rec.": 2, "RCD": 2, "Jaccard": 2, "MTPO [ms]": 3, "MTTD": 1
        })
        summary = summary.sort_values(by=sort_by)
        summary.drop(["Jaccard", "Parameters", "Dims"], axis=1, inplace=True)
        summary["PC"] = summary["PC"].astype(int)
        print(summary.set_index(["Dataset", "Approach"]).to_latex(escape=False))