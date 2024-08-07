import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from changeds.metrics import fb_score, true_positives, false_positives, precision, recall, false_negatives
from scipy.stats import spearmanr
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from E_synthetic_data import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, \
    get_abcd_hyperparameters_from_str, cm2inch

sns.set_theme(context="paper", style="ticks", palette="deep")

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def compute_region_metrics(df: pd.DataFrame, thresh: float = 2.2):
    if not np.any(pd.isnull(df["dims-found"]) == False):
        return np.nan, np.nan, np.nan
    changes = df["change-point"]
    idxs = [i for i, change in enumerate(changes) if change]
    regions_gt = df["dims-gt"].iloc[idxs]
    all_dims = df["ndims"].iloc[idxs]
    if "ABCD" in df["approach"].iloc[0]:
        regions_detected = df["dims-p"].iloc[idxs]
        regions_detected = [str_to_arr(s, float) for s in regions_detected]
        regions_detected = [[j for j in range(len(row)) if row[j] < thresh]
                            for row in regions_detected]
    else:
        regions_detected = df["dims-found"].iloc[idxs]
    prec_scores = []
    recall_scores = []
    acc_scores = []
    for gt, found, dims in zip(regions_gt, regions_detected, all_dims):
        try:
            gt = str_to_arr(gt, int)
            if not "ABCD" in df["approach"].iloc[0]:
                found = str_to_arr(found, int)
            full_space = np.arange(dims)
            tp = len(np.intersect1d(gt, found))
            fp = len(np.setdiff1d(found, np.intersect1d(gt, found)))
            fn = len(np.setdiff1d(gt, np.intersect1d(gt, found)))
            tn = len(np.intersect1d(np.setdiff1d(full_space, gt),
                                    np.setdiff1d(full_space, found)))
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            acc = (tp + tn) / (tp + tn + fp + fn)
            prec_scores.append(prec)
            recall_scores.append(rec)
            acc_scores.append(acc)
        except:
            continue
    prec = np.nanmean(prec_scores) if len(prec_scores) > 0 else np.nan
    rec = np.nanmean(recall_scores) if len(recall_scores) > 0 else np.nan
    acc = np.nanmean(acc_scores) if len(acc_scores) > 0 else np.nan
    return prec, rec, acc


def compute_severity_metric(df: pd.DataFrame):
    df = df[df["change-point"].astype(bool)]
    severities = df["severity-gt"]
    detected_severities = df["severity"]
    x = severities.to_numpy()
    y = detected_severities.to_numpy()
    if np.all(pd.isna(x)):
        return np.nan, 0
    na_indices = [i for i in range(len(x)) if pd.isna(x[i])]
    x = np.delete(x, na_indices)
    y = np.delete(y, na_indices)
    if len(x) < 2 or len(y) < 2:
        return np.nan, 0
    try:
        corr, p = spearmanr(x, y)
    except:
        x = np.array([float(x_i[1:-1]) for x_i in x.flatten()])
        y = y.flatten().astype(float)
        corr, p = spearmanr(x, y)
    return corr, len(y) - 1


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
            if not file.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",").convert_dtypes()
            df = fill_df(df)
            approach = np.unique(df["approach"])[0]
            params = np.unique(df["parameters"])[0]
            dataset = np.unique(df["dataset"])[0]
            dims = np.unique(df["ndims"])[0]
            # if not "(pca)" in approach:
            #     continue
            if "ABCD" in approach or "ABCD0" in approach:
                parsed_params = get_abcd_hyperparameters_from_str(params)
                E, eta = parsed_params[1], parsed_params[2]
            else:
                E = np.nan
                eta = np.nan
            # result_df.append([np.nan, dataset, dims, approach, params, E, eta,
            #                   np.nan, np.nan, np.nan, np.nan, severity, weight, np.nan])
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
                region_prec, region_rec, acc = compute_region_metrics(rep_data, thresh=2.5)
                severity, weight = compute_severity_metric(rep_data)
                f1_region = 2 * (region_rec * region_prec) / (region_rec + region_prec)
                result_df.append([
                    rep, dataset, dims, approach, params, E, eta, f1_region, region_prec, region_rec, severity, weight,
                    f1, acc
                ])
        result_df = pd.DataFrame(result_df, columns=["Rep", "Dataset", "Dims", "Approach", "Parameters", "E", "eta",
                                                     "F1 (Subspace)", "Prec. (Region)",
                                                     "Rec. (Region)", r"Spearman $\rho$", "Weight", "F1", "SAcc."])
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), mode="w+", index=False)
    if "Pearson R" in result_df.columns:
        result_df.rename({r"Spearman $\rho$": r"Spearman $\rho$"}, axis=1, inplace=True)
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), mode="w+", index=False)
    result_df = result_df[result_df["Dataset"] != "RBF"]
    for _, gdf in result_df.groupby(["Approach", "Dataset", "Dims"]):
        index = gdf.index
        gdf = gdf[r"Spearman $\rho$"].fillna(method="ffill")
        result_df[r"Spearman $\rho$"].loc[index] = gdf
    result_df = result_df.groupby(["Dataset", "Approach", "Parameters", "Dims"]).mean().reset_index()
    sort_by = ["Dataset", "Dims", "Approach"]
    result_df = result_df.sort_values(by=sort_by)
    print(np.all(np.isnan(result_df[r"Spearman $\rho$"])))
    result_df["F1 (Subspace)"][result_df["F1 (Subspace)"].isna()] = 0
    result_df[r"Spearman $\rho$"][result_df[r"Spearman $\rho$"].isna()] = 0
    result_df["SAcc."][result_df["SAcc."].isna()] = 0
    print((result_df
           .groupby(["Approach"])
           .mean()
           .reset_index()
           .round(decimals={"F1 (Subspace)": 2,
                            "Prec. (Region)": 2,
                            "Rec. (Region)": 2,
                            r"Spearman $\rho$": 2})
           .reset_index()
           .to_markdown())
          )
    result_df = result_df.sort_values(by=["Dims", "Dataset"]).astype(dtype={"Dims": str})
    result_df[r"Spearman $\rho$"][result_df["Approach"] == "D3"] = result_df[result_df["Approach"] == "D3"][
        r"Spearman $\rho$"]
    result_df = result_df[result_df["Dataset"] != "Average"]
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD (ae)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "ABCD (pca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "ABCD (kpca)"
    result_df = result_df.fillna(0.0)
    result_df = result_df.sort_values(by="Approach")
    n_colors = len(np.unique(result_df["Approach"]))
    melted_df = pd.melt(result_df, id_vars=["Dataset", "Approach", "Parameters", "E", "eta", "Dims"],
                        value_vars=["F1", "SAcc.", r"Spearman $\rho$"],
                        var_name="Metric", value_name="Value")
    melted_df = melted_df[melted_df["Value"].isna() == False]
    melted_df[r"$\eta$"] = melted_df["eta"]
    g = sns.catplot(data=melted_df, x="Approach", y="Value", col="Dataset", row="Metric", kind="box",
                    linewidth=0.7, fliersize=2, sharey="row", showfliers=False,
                    )
    g.set(xlabel=None)
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.axhline(0.0, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axhline(0.25, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axhline(0.5, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axhline(0.75, color="gray", lw=0.7, ls="--", zorder=0)
        ax.axhline(1.0, color="gray", lw=0.7, ls="--", zorder=0)
        if i == 0:
            ax.set_ylabel("F1")
        if i == 4:
            ax.set_ylabel("SAcc.")
        if i == 8:
            ax.set_ylabel(r"Spearman $\rho$")
        if i >= 8:
            ax.set_ylim(bottom=-0.3, top=0.8)
        if i < 4:
            col_title = ax.get_title().split(" = ")[-1]
            if col_title.startswith("Normal"):
                a, b = col_title.split("al")
                col_title = a + "." + b
            ax.set_title(col_title)
        else:
            ax.set_title("")
    plt.gcf().set_size_inches(cm2inch(16, 7.5))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(wspace=.1)
    plt.savefig(os.path.join("..", "figures", "evaluation_drift_region.pdf"))
    plt.show()

    melted_df = melted_df[melted_df["Metric"] != "F1"]
    g = sns.catplot(data=melted_df[melted_df["Approach"] != "Average"],
                    x="Approach", y="Value", row="Metric", kind="box",
                    linewidth=0.7, fliersize=2, sharey="row", showfliers=False,
                    )
    g.set(xlabel=None)
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_ylabel("")
        col_title = ax.get_title().split(" = ")[-1]
        if col_title.startswith("Normal"):
            a, b = col_title.split("al")
            col_title = a + "." + b
        if col_title == r"Spearman $\rho$":
            col_title = r"Spearman correlation with change severity"
        if col_title == "SAcc.":
            col_title = "Accuracy at detecting change subspace"
        ax.set_title(col_title)
    plt.xticks(rotation=25, ha='right')
    plt.gcf().set_size_inches(3.3, 2.3)
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(left=0.15, wspace=0.1, hspace=0.5, right=0.98)
    plt.savefig(os.path.join("..", "figures", "evaluation_drift_region_presentation.pdf"))
    plt.show()

    melted_df = melted_df[melted_df["Metric"] != "F1"]
    abcd = np.logical_or(melted_df["Approach"] == "ABCD (ae)",
                         melted_df["Approach"] == "ABCD (pca)")
    abcd = np.logical_or(abcd, melted_df["Approach"] == "ABCD (kpca)")
    abcd = melted_df[abcd]
    abcd["Approach"][abcd["Approach"] == "ABCD (ae)"] = "ae"
    abcd["Approach"][abcd["Approach"] == "ABCD (kpca)"] = "kpca"
    abcd["Approach"][abcd["Approach"] == "ABCD (pca)"] = "pca"
    abcd = abcd.groupby(["Approach", "Parameters", "Dataset", "Metric", "Dims"]).mean().reset_index()
    abcd_eta = abcd.copy()
    average = abcd_eta.copy()
    average["Dataset"] = "Average"
    abcd_eta = abcd_eta.sort_values(by=["Dims", "Metric"])
    abcd_eta = pd.concat([average, abcd_eta], axis=0)
    g = sns.catplot(x=r"$\eta$", col="Dataset", y="Value", row="Metric", hue="Approach", errwidth=1,
                    data=abcd_eta, kind="bar",
                    legend=False,
                    height=cm2inch(2.5)[0], aspect=1.5, sharey="row")
    axes = plt.gcf().axes
    plt.gcf().subplots_adjust(left=0.08)
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel("SAcc.")
        if i == 5:
            ax.set_ylabel(r"Spearman $\rho$")
        current_title = ax.get_title()
        dataset = current_title.split(" = ")[-1]
        if "Norm" in dataset:
            dataset = "Norm.-" + dataset.split("-")[-1]
        ax.set_title(dataset)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i < 5:
            ax.axhline(0.5, lw=0.7, ls="--", color="gray", zorder=0)
            ax.axhline(1.0, lw=0.7, ls="--", color="gray", zorder=0)
        if i >= 5:
            ax.set_title("")
            ax.axhline(0.25, lw=0.7, ls="--", color="gray", zorder=0)
            ax.axhline(.5, lw=0.7, ls="--", color="gray", zorder=0)
            ax.set_ylim(bottom=0.0 if i < 5 else 0, top=0.7)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "sensitivity-study-eta-region-severity.pdf"))
    plt.show()
