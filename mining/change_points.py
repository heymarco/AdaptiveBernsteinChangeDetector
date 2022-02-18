import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from changeds.metrics import *
from util import get_last_experiment_dir, str_to_arr


def compute_jaccard(df: pd.DataFrame):
    if not np.any(pd.isnull(df["dims-found"]) == False):
        return np.nan
    changes = df["change-point"]
    idxs = [i for i, change in enumerate(changes) if change]
    regions_gt = df["dims-gt"].iloc[idxs]
    regions_detected = df["dims-found"].iloc[idxs]
    results = []
    for a, b in zip(regions_gt, regions_detected):
        print(a, b)
        a = str_to_arr(a, np.int)
        b = str_to_arr(b, np.int)
        jac = jaccard(a, b) if len(b) > 0 else np.nan
        results.append(jac)
    return np.mean(results)


def mean_time_per_example(df):
    series = df["time"]
    return series.diff().mean()


if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir()
    all_files = os.listdir(last_exp_dir)
    result_df = []
    for file in tqdm(all_files):
        df = pd.read_csv(os.path.join(last_exp_dir, file)).convert_dtypes().ffill()
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
            f1 = fb_score(true_cps, reported_cps, T=3000)
            mttd = mean_until_detection(true_cps, reported_cps)
            jac = compute_jaccard(rep_data)
            mtpe = mean_time_per_example(rep_data)
            mtpe = round(mtpe / 10e6, 3)
            result_df.append([
                dataset, approach, params, prec, rec, f1, mttd, jac, mtpe
            ])
    result_df = pd.DataFrame(result_df, columns=["Dataset", "Approach", "Parameters",
                                                 "Prec.", "Rec.", "F1", "MTTD", "Jaccard", "MTPE [ms]"])
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters"]).mean().round(2)
    result_df.reset_index()
    print(result_df.sort_values(by=["Dataset", "Approach", "Parameters"]).to_latex(escape=False))
