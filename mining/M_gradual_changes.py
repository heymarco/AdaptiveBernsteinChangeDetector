import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from tqdm import tqdm
import seaborn as sns

from changeds.metrics import *
from util import get_last_experiment_dir, fill_df, create_cache_dir_if_needed, \
    get_abcd_hyperparameters_from_str, change_bar_width, cm2inch
from E_gradual_changes import ename

sns.set_theme(context="paper", style="ticks", palette="deep")
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


def get_result_df(sort_by) -> pd.DataFrame:
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    cache_dir = create_cache_dir_if_needed(last_exp_dir)
    if os.path.exists(os.path.join(cache_dir, "cached.csv")):
        result_df = pd.read_csv(os.path.join(cache_dir, "cached.csv"))
    else:
        result_df = compute_metrics(all_files, cache_dir, last_exp_dir)

    result_df = result_df[
        ["Dataset", "Dims", "Approach", "Parameters",
         "F1", "Prec.", "Rec.",
         "MTD"
         ]]

    result_df = (result_df
                 .sort_values(by=sort_by)
                 .apply(func=add_params_to_df, axis=1))
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD (ae)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "ABCD (pca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "ABCD (kpca)"
    return result_df


def add_params_to_df(row):
    if "ABCD" not in row["Approach"]:
        return row
    params = get_abcd_hyperparameters_from_str(row["Parameters"])
    parsed = params
    row[r"$\eta$"] = parsed[2]
    row["E"] = parsed[1]
    return row


def create_boxplots_for_approaches(result_df):
    # result_df = result_df[np.logical_and(
    #     result_df["Approach"] == "D3", result_df["Parameters"].apply(lambda x: "dt" in x)
    # ) == False]
    result_df = result_df.groupby(["Approach", "Parameters", "Dataset"]).mean().reset_index()
    mean = result_df.groupby(["Approach", "Parameters", "Dataset"]).mean().reset_index()
    mean["Dataset"] = "Average"
    result_df = pd.concat([mean, result_df])
    result_df["MTD (thousands)"] = result_df["MTD"] / 1000
    value_vars = ["F1", "Prec.", "Rec.", "MTD (thousands)"]
    melted_df = pd.melt(result_df, id_vars=["Dataset", "Approach"], value_vars=value_vars,
                        var_name="Metric", value_name="Value")
    melted_df = melted_df[melted_df["Value"].isna() == False]
    g = sns.catplot(data=melted_df, x="Approach", y="Value", col="Dataset", row="Metric", kind="box",
                    linewidth=0.7, fliersize=2, sharey="row",
                    )
    g.set(xlabel=None)
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i < 8:
            col_title = ax.get_title().split(" = ")[-1]
            if col_title.startswith("Normal"):
                a, b = col_title.split("al")
                col_title = a + "." + b
            ax.set_title(col_title)
        else:
            ax.set_title("")
        if i % 8 == 0:
            ax.set_ylabel(value_vars[int(i / 8)])
        if i >= 3 * 8:
            ax.axhline(1e0, color="gray", lw=0.7, ls="--", zorder=0)
            ax.axhline(1e-1, color="gray", lw=0.7, ls="--", zorder=0)
            ax.axhline(1e-2, color="gray", lw=0.7, ls="--", zorder=0)
            ax.set_yscale("log")
        else:
            ax.axhline(0.0, color="gray", lw=0.7, ls="--", zorder=0)
            ax.axhline(0.25, color="gray", lw=0.7, ls="--", zorder=0)
            ax.axhline(0.5, color="gray", lw=0.7, ls="--", zorder=0)
            ax.axhline(0.75, color="gray", lw=0.7, ls="--", zorder=0)
            ax.axhline(1.0, color="gray", lw=0.7, ls="--", zorder=0)
    plt.gcf().set_size_inches(3.5 * 2.5, 3.5 * 1.5)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05)
    plt.savefig(os.path.join("..", "figures", "evaluation_gradual_changes.pdf"))
    plt.show()

    g = sns.catplot(data=melted_df[melted_df["Approach"] != "Average"],
                    x="Approach", y="Value", row="Metric", kind="box",
                    linewidth=0.7, showfliers=False, sharey="row",
                    )
    g.set(xlabel=None)
    for i, ax in enumerate(plt.gcf().axes):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        col_title = ax.get_title().split(" = ")[-1]
        if col_title.startswith("Normal"):
            a, b = col_title.split("al")
            col_title = a + "." + b
        ax.set_title(col_title)
        ax.set_ylabel("")  # (value_vars[i])
    plt.xticks(rotation=25, ha='right')
    plt.gcf().set_size_inches(3.3, 3.3)
    plt.tight_layout(pad=.2)
    plt.subplots_adjust(wspace=0.05, hspace=.7)
    plt.savefig(os.path.join("..", "figures", "evaluation_gradual_changes_presentation.pdf"))
    plt.show()


def E_sensitivity_plot(result_df):
    abcd = np.logical_or(result_df["Approach"] == "ABCD (ae)",
                         result_df["Approach"] == "ABCD (pca)")
    abcd = np.logical_or(abcd, result_df["Approach"] == "ABCD (kpca)")
    abcd = result_df[abcd]
    abcd[r"$E$"] = abcd["E"].astype(int)
    abcd["Approach"][abcd["Approach"] == "ABCD (ae)"] = "ae"
    abcd = abcd.groupby(["Approach", r"Parameters", "Dataset"]).mean().reset_index()
    abcd[r"$E$"] = abcd[r"$E$"].astype(int)
    abcd_E = abcd[abcd["Approach"] == "ae"].copy()
    average = abcd_E.copy()  # '.groupby(["Approach", "Parameters", r"$E$"]).mean().reset_index()
    average["Dataset"] = "Average"
    abcd_E = abcd_E.sort_values(by="Dims")
    abcd_E = pd.concat([average, abcd_E], axis=0)
    n_colors = len(np.unique(abcd_E["E"]))
    ax = sns.barplot(x="Dataset", y="F1", hue="E", data=abcd_E, palette=sns.cubehelix_palette(n_colors=n_colors),
                     errwidth=1)
    ax.set_xlabel("")
    ax.axhline(0.5, color="gray", lw=0.7, ls="--", zorder=0)
    ax.axhline(0.75, color="gray", lw=0.7, ls="--", zorder=0)
    ax.axhline(1, color="gray", lw=0.7, ls="--", zorder=0)
    ax.set_ylim(bottom=.3)
    ax.legend(title=r"$E$", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    change_bar_width(ax, .23)
    plt.gcf().set_size_inches(cm2inch(16, 2.5))
    plt.tight_layout(pad=.5)
    plt.subplots_adjust(right=0.85)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "sensitivity-study-E.pdf"))
    plt.show()


def eta_sensitivity_plot(result_df):
    abcd = np.logical_or(result_df["Approach"] == "ABCD (ae)",
                         result_df["Approach"] == "ABCD (pca)")
    abcd = np.logical_or(abcd, result_df["Approach"] == "ABCD (kpca)")
    abcd = result_df[abcd]
    abcd[r"$E$"] = abcd["E"].astype(int)
    abcd["Approach"][abcd["Approach"] == "ABCD (ae)"] = "ae"
    abcd["Approach"][abcd["Approach"] == "ABCD (kpca)"] = "kpca"
    abcd["Approach"][abcd["Approach"] == "ABCD (pca)"] = "pca"
    abcd = abcd.groupby(["Approach", r"Parameters", "Dataset"]).mean().reset_index()
    abcd_eta = abcd.copy()
    average = abcd_eta.copy()
    average["Dataset"] = "Average"
    abcd_eta = abcd.sort_values(by="Dims").copy()
    abcd_eta = pd.concat([average, abcd_eta], axis=0)

    print(abcd_eta[np.logical_and(
        abcd_eta["Dataset"] == "Gas",
        abcd_eta[r"$\eta$"] == "0.7"
    )])

    g = sns.catplot(x=r"$\eta$", y="F1", col="Dataset", hue="Approach", errwidth=1,
                    data=abcd_eta, kind="bar",
                    height=cm2inch(5)[0], aspect=0.4, sharex=False)
    axes = g.axes.flatten()
    for i, ax in enumerate(axes):
        ax.axhline(0.25, lw=0.7, ls="--", color="gray", zorder=0)
        ax.axhline(0.75, lw=0.7, ls="--", color="gray", zorder=0)
        ax.axhline(.5, lw=0.7, ls="--", color="gray", zorder=0)
        ax.axhline(1, lw=0.7, ls="--", color="gray", zorder=0)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        if i < 8:
            current_title = ax.get_title()
            dataset = current_title.split(" = ")[-1]
            ax.set_title(dataset)
        else:
            ax.set_title("")
        ax.set_xticks(ax.get_xticks(), [])
        ax.set_xticklabels([0.3, 0.5, 0.7])
    sns.move_legend(g, "upper center", ncol=3, title=None, frameon=False)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(top=0.66)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", "sensitivity-study-eta.pdf"))
    plt.show()


def print_latex_table(result_df, sort_by):
    summary = (result_df.copy()
               .groupby(["Dataset", "Approach", "Parameters"])
               .mean()
               .reset_index()
               .drop(["E", r"$\eta$"], axis=1))
    summary = filter_best(summary)
    summary = summary.round(decimals={
        "F1": 2,
        "Prec.": 2,
        "Rec.": 2,
        "MTD": 1
    })
    summary.drop(["Parameters", "Dims"], axis=1, inplace=True)
    print(summary.set_index(["Dataset", "Approach"]).to_latex(escape=False))


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
            true_cps = [i for i in rep_data.index if rep_data["is-change"].loc[i]]
            cp_distance = 2000
            reported_cps = [i for i in rep_data.index if rep_data["change-point"].loc[i]]
            tp = true_positives(true_cps, reported_cps, cp_distance)
            fp = false_positives(true_cps, reported_cps, cp_distance)
            fn = false_negatives(true_cps, reported_cps, cp_distance)
            n_seen_changes = len(true_cps)
            delays = rep_data["delay"].loc[reported_cps].tolist()
            prec = precision(tp, fp, fn)
            rec = recall(tp, fp, fn)
            f1 = 2 * (prec * rec) / (prec + rec) if prec > 0 or rec > 0 else 0.0
            MTD = mean_until_detection(true_cps, reported_cps)
            mae_delay = mean_cp_detection_time_error(true_cps, reported_cps,
                                                     delays) if "ABCD" in approach else np.nan
            result_df.append([
                dataset, dims, approach, params, rep,
                f1, prec, rec,
                MTD,
                n_seen_changes, mae_delay
            ])
    result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "Parameters", "rep", "F1", "Prec.", "Rec.", "MTD",
                                                 # "MTPO [ms]",
                                                 "PC", "CP-MAE"])
    result_df = result_df.fillna({"F1": 0.0, "Prec.": 0.0, "Rec.": 0.0})
    result_df.loc[result_df["F1"] == 0, "MTD"] = np.nan
    result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    return result_df


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def filter_best(df, add_mean: bool = True):
    indices = []
    df = df.groupby(["Dataset", "Approach", "Parameters"]).mean().reset_index()
    for _, gdf in df.groupby(["Dataset", "Approach"]):
        gdf = gdf.fillna({"F1": 0.0}).sort_values(by=["F1", "MTD"])
        max_index = gdf["F1"].idxmax()
        indices.append(max_index)
    indices = np.unique(indices)
    df = (df.loc[indices]
          .fillna(0.0))
    if add_mean:
        df = add_mean_column(df)
    return df.reset_index().sort_values(by=["Dataset", "Approach"])


if __name__ == '__main__':
    print_summary = True
    plot_E_sensitivity_study = True
    plot_eta_sensitivity_study = True
    plot_gradual_changes_comparison = True

    sort_by = ["Dims", "Dataset", "Approach", "Parameters"]
    result_df = get_result_df(sort_by)

    if print_summary:
        print_latex_table(result_df.copy(), sort_by)

    if plot_eta_sensitivity_study:
        eta_sensitivity_plot(result_df.copy())

    if plot_E_sensitivity_study:
        E_sensitivity_plot(result_df.copy())

    if plot_gradual_changes_comparison:
        create_boxplots_for_approaches(result_df.copy())