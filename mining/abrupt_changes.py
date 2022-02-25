import os

import pandas as pd
from tqdm import tqdm

from changeds.metrics import *
from util import get_last_experiment_dir, str_to_arr
from E_abrupt_changes import ename


def compute_jaccard(df: pd.DataFrame):
    if not np.any(pd.isnull(df["dims-found"]) == False):
        return np.nan
    changes = df["change-point"]
    idxs = [i for i, change in enumerate(changes) if change]
    regions_gt = df["dims-gt"].iloc[idxs]
    regions_detected = df["dims-found"].iloc[idxs]
    results = []
    for a, b in zip(regions_gt, regions_detected):
        try:
            a = str_to_arr(a, int)
            b = str_to_arr(b, int)
            jac = jaccard(a, b) if len(b) > 0 else np.nan
            results.append(jac)
        except:
            continue
    return np.nanmean(results) if len(results) > 0 else np.nan


def mean_time_per_example(df):
    series = df["time"]
    return series.diff().mean()


def compare(print_summary: bool, summary_kwargs={"worst": False, "median": True}):
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    result_df = []
    j = 0
    for file in tqdm(all_files):
        j += 1
        # if j < 270 or j > 280:
        #     continue
        df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",").convert_dtypes().ffill()
        approach = np.unique(df["approach"])[0]
        params = np.unique(df["parameters"])[0]
        dataset = np.unique(df["dataset"])[0]
        for rep, rep_data in df.groupby("rep"):
            true_cps = [i for i in range(len(rep_data)) if rep_data["is-change"].iloc[i]]
            cp_distance = true_cps[0]
            reported_cps = [i for i in range(len(rep_data)) if rep_data["change-point"].iloc[i]]
            tp = true_positives(true_cps, reported_cps, cp_distance)
            fp = false_positives(true_cps, reported_cps, cp_distance)
            fn = false_negatives(true_cps, reported_cps, cp_distance)
            prec = precision(tp, fp, fn)
            rec = recall(tp, fp, fn)
            f1 = fb_score(true_cps, reported_cps, T=2000)
            mttd = mean_until_detection(true_cps, reported_cps)
            jac = compute_jaccard(rep_data)
            mtpe = mean_time_per_example(rep_data)
            mtpe = mtpe / 10e6
            rcd = ratio_changes_detected(true_cps, reported_cps)
            result_df.append([
                dataset, approach, params, f1, prec, rec, rcd,
                jac, mttd, mtpe
            ])
    result_df = pd.DataFrame(result_df, columns=["Dataset", "Approach", "Parameters", "F1", "Prec.",
                                                 "Rec.", "RCD", "Jaccard", "MTTD", "MTPO [ms]"])
    if print_summary:
        summary = filter_best(result_df, **summary_kwargs)
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters"]).mean()
    result_df = result_df.round(decimals={
        "F1": 2, "Prec.": 2, "Rec.": 2, "RCD": 2, "Jaccard": 2, "MTPO [ms]": 3, "MTTD": 1
    })
    result_df = result_df.sort_values(by=["Dataset", "Approach", "Parameters"])
    result_df.drop(["Jaccard", "Parameters"], axis=1, inplace=True)
    print(result_df.to_latex(escape=False))
    if print_summary:
        summary = summary.round(decimals={
            "F1": 2, "Prec.": 2, "Rec.": 2, "RCD": 2, "Jaccard": 2, "MTPO [ms]": 3, "MTTD": 1
        })
        summary.drop(["Jaccard", "Parameters"], axis=1, inplace=True)
        print(summary.to_latex(escape=False))


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def filter_best(df, worst: bool, median: bool, add_mean: bool = True):
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
    if add_mean:
        df = add_mean_column(df)
    return df


if __name__ == '__main__':
    compare(print_summary=True)
