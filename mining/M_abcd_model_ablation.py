import os
from tqdm import tqdm
import seaborn as sns

from changeds.metrics import *
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, get_abcd_hyperparameters_from_str
from E_abcd_model_ablation import ename

sns.set_theme(context="paper", style="ticks", palette="deep")
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
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
    if "ABCD" not in row["Approach"]:
        return row
    params = get_abcd_hyperparameters_from_str(row["Parameters"])
    parsed = params
    row[r"$\eta$"] = parsed[2]
    row["E"] = parsed[1]
    return row


def compute_metrics(all_files, cache_dir, last_exp_dir):
    result_df = []
    j = 0
    for file in tqdm(all_files):
        if not file.endswith(".csv"):
            continue
        j += 1
        df = pd.read_csv(os.path.join(last_exp_dir, file), index_col=0, sep=",").convert_dtypes()
        df = fill_df(df)
        df["ndims"].fillna(0, inplace=True)
        approach = np.unique(df["approach"])[0]
        params = np.unique(df["parameters"])[0]
        dataset = np.unique(df["dataset"])[0]
        dims = np.unique(df["ndims"])[0]
        for rep, rep_data in df.groupby("rep"):
            true_cps = [i for i in rep_data.index if rep_data["is-change"].loc[i]]  # TODO: check if this is correct
            cp_distance = 2000
            reported_cps = [i for i in rep_data.index if rep_data["change-point"].loc[i]]
            tp = true_positives(true_cps, reported_cps, cp_distance)
            fp = false_positives(true_cps, reported_cps, cp_distance)
            fn = false_negatives(true_cps, reported_cps, cp_distance)
            n_seen_changes = len(true_cps)
            delays = rep_data["delay"].loc[reported_cps].tolist()
            prec = precision(tp, fp, fn)
            rec = recall(tp, fp, fn)
            f1 = fb_score(true_cps, reported_cps, T=cp_distance)
            f05 = fb_score(true_cps, reported_cps, T=cp_distance, beta=0.5)
            f2 = fb_score(true_cps, reported_cps, T=cp_distance, beta=2)
            MTD = mean_until_detection(true_cps, reported_cps)
            mae_delay = mean_cp_detection_time_error(true_cps, reported_cps,
                                                     delays) if "ABCD" in approach else np.nan
            result_df.append([
                dataset, dims, approach, params, rep, f1, f05, f2, prec, rec,
                MTD,
                n_seen_changes, mae_delay
            ])
    result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "Parameters", "rep", "F1", "F0.5",
                                                 "F2", "Prec.", "Rec.", "MTD",
                                                 # "MTPO [ms]",
                                                 "PC", "CP-MAE"])
    result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    return result_df


def mean_time_per_example(df):
    delta_t = df["time"].iloc[-1] - df["time"].iloc[1]
    delta_obs = df.index[-1] - df.index[1]
    return delta_t / delta_obs


def plot_window_type_ablation(result_df):
    result_df["Approach"][result_df["Approach"] == "FRW (ae)"] = "RW (ae)"
    result_df["Approach"][result_df["Approach"] == "FRW (kpca)"] = "RW (kpca)"
    result_df["Approach"][result_df["Approach"] == "FRW (pca)"] = "RW (pca)"
    avg_df = result_df.groupby(["Approach", "Dataset"]).mean().reset_index()
    avg_df["Window"] = ""
    avg_df["Model"] = ""
    for index, row in avg_df.iterrows():
        approach = row["Approach"]
        alg, model = approach.split(" ")
        model = model[1:-1]
        # alg = "RW" if alg == "FRW" else alg
        row["Window"] = "AW" if alg == "ABCD" else alg
        row["Model"] = model.upper()
        avg_df.loc[index] = row
    avg_df = avg_df[["Model", "Window", "F1", "Prec.", "Rec.", "MTD"]].sort_values(["Model", "Window"]).groupby(
        ["Model", "Window"]).median()
    print(avg_df.round(decimals={
        "F1": 2, "Prec.": 2, "Rec.": 2, "MTD": 1
    }).to_latex(escape=False))  # (escape=False))
    del avg_df


if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    if os.path.exists(os.path.join(cache_dir, "cached.csv")):
        result_df = pd.read_csv(os.path.join(cache_dir, "cached.csv"))
    else:
        result_df = compute_metrics(all_files, cache_dir, last_exp_dir)

    result_df = result_df[
        ["Dataset", "Dims", "Approach", "Parameters",
         "F0.5", "F1", "F2", "Prec.", "Rec.",
         "MTD"
         ]]

    sort_by = ["Dims", "Dataset", "Approach", "Parameters"]
    result_df = (result_df
                 .sort_values(by=sort_by)
                 .apply(func=add_params_to_df, axis=1))
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD (ae)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "ABCD (pca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "ABCD (kpca)"

    plot_window_type_ablation(result_df)
