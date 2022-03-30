import os

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

from changeds.metrics import *
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, \
    get_abcd_hyperparameters_from_str
from E_gradual_changes import ename

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


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


def add_params_to_df(row):
    if row["Approach"] != "ABCD2":
        return row
    params = get_abcd_hyperparameters_from_str(row["Parameters"])
    _, E, eta, _ = params
    row[r"$\eta$"] = eta
    row["E"] = E
    return row


def mean_time_per_example(df):
    delta_t = df["time"].iloc[-1] - df["time"].iloc[1]
    delta_obs = df.index[-1] - df.index[1]
    return delta_t / delta_obs


def compare(print_summary: bool, summary_kwargs={"worst": False, "median": True}):
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    df_cpy = False
    if os.path.exists(os.path.join(cache_dir, "cached.csv")):
        result_df = pd.read_csv(os.path.join(cache_dir, "cached.csv"))
    else:
        result_df = []
        j = 0
        for file in tqdm(all_files):
            if not file.endswith(".csv"):
                continue
            j += 1
            # if j > 2:
            #     break
            df = pd.read_csv(os.path.join(last_exp_dir, file), index_col=0, sep=",").convert_dtypes()
            df = fill_df(df)
            approach = np.unique(df["approach"])[0]
            params = np.unique(df["parameters"])[0]
            dataset = np.unique(df["dataset"])[0]
            dims = np.unique(df["ndims"])[0]
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
                mttd = mean_until_detection(true_cps, reported_cps)
                mae_delay = mean_cp_detection_time_error(true_cps, reported_cps,
                                                         delays) if "ABCD" in approach else np.nan
                mtpe = mean_time_per_example(rep_data)
                mtpe = mtpe / 10e6
                rcd = ratio_changes_detected(true_cps, reported_cps)
                result_df.append([
                    dataset, dims, approach, params, rep, f1, prec, rec, rcd,
                    mttd, mtpe, n_seen_changes, mae_delay
                ])
        result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "Parameters", "rep", "F1", "Prec.",
                                                     "Rec.", "RCD", "MTTD", "MTPO [ms]", "PC", "CP-MAE"])
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    result_df = result_df[["Dataset", "Dims", "Approach", "Parameters", "F1", "Prec.", "Rec.", "RCD", "MTTD", "MTPO [ms]"]]
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters"]).mean().reset_index()
    if print_summary:
        summary = filter_best(result_df, **summary_kwargs)
    result_df = result_df.round(decimals={
        "F1": 2, "Prec.": 2, "Rec.": 2, "RCD": 2, "MTPO [ms]": 3, "MTTD": 1
    })
    result_df[result_df["Dataset"] == "Average"] = 0
    sort_by = ["Dims", "Dataset", "Approach", "Parameters"]
    result_df = result_df.sort_values(by=sort_by)
    result_df = result_df.apply(func=add_params_to_df, axis=1)
    if print_summary:
        summary = summary.round(decimals={
            "F1": 2, "Prec.": 2, "Rec.": 2, "RCD": 2, "MTPO [ms]": 3, "MTTD": 1
        })
        summary = summary.sort_values(by=sort_by)
        summary.drop(["Parameters", "Dims"], axis=1, inplace=True)
        print(summary.set_index(["Dataset", "Approach"]).to_latex(escape=False))
    abcd = result_df[result_df["Approach"] == "ABCD2"]
    abcd["E"] = abcd["E"].astype(int)
    # average = abcd.groupby(["Approach", "E", r"$\eta$"]).mean().reset_index()
    # average["Dataset"] = "Average"
    # abcd = pd.concat([abcd, average], axis=0)
    g = sns.catplot(x="E", y="F1",
                    hue=r"$\eta$", col="Dataset",
                    data=abcd, kind="point", palette=sns.cubehelix_palette(n_colors=3),
                    height=2, aspect=.43, errwidth=2, scale=0.5)
    axes = plt.gcf().axes
    for ax in axes:
        current_title = ax.get_title()
        dataset = current_title.split(" = ")[1]
        ax.set_title(dataset)
    plt.gcf().subplots_adjust(left=0.08)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "sensitivity-study.pdf"))
    plt.show()


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def filter_best(df, worst: bool, median: bool, add_mean: bool = True):
    df = df.sort_values(by=["F1", "MTTD"])
    median_indices = []
    min_indices = []
    max_indices = []
    indices = []
    df = df.groupby(["Dataset", "Approach", "Parameters"]).mean().reset_index()
    for _, gdf in df.groupby(["Dataset", "Approach"]):
        gdf = gdf.dropna()
        if len(gdf) == 0:
            continue
        max_index = gdf["F1"].idxmax()
        indices.append(max_index)
        if "ABCD2" in gdf["Approach"].to_numpy():
            max_indices.append(max_index)
            if median:
                med = gdf["F1"].dropna().median()
                median_index = (gdf["F1"] - med).abs().idxmin()
                if median_index == max_index and len(gdf) > 1:  # second clause should be particularly relevant for testing.
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


if __name__ == '__main__':
    compare(print_summary=True)
