import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from tqdm import tqdm
import seaborn as sns

from changeds.metrics import *
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, \
    get_abcd_hyperparameters_from_str, change_bar_width
from E_abcd_model_ablation import ename

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


def mean_time_per_example(df):
    delta_t = df["time"].iloc[-1] - df["time"].iloc[1]
    delta_obs = df.index[-1] - df.index[1]
    return delta_t / delta_obs


def compare(print_summary: bool,
            plot_eta_sensitivity_study: bool = False,
            plot_E_sensitivity_study: bool = False,
            plot_gradual_changes_comparison: bool = False, summary_kwargs={"worst": False, "median": True}):
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
                mtpe = mean_time_per_example(rep_data)
                mtpe = mtpe / 10e6
                result_df.append([
                    dataset, dims, approach, params, rep, f1, f05, f2, prec, rec,
                    MTD, mtpe, n_seen_changes, mae_delay
                ])
        result_df = pd.DataFrame(result_df, columns=["Dataset", "Dims", "Approach", "Parameters", "rep", "F1", "F0.5",
                                                     "F2", "Prec.", "Rec.", "MTD",
                                                     "MTPO [ms]", "PC", "CP-MAE"])
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    result_df = result_df[["Dataset", "Dims", "Approach", "Parameters", "F0.5", "F1", "F2", "Prec.", "Rec.",
                           "MTD", "MTPO [ms]"]]
    if print_summary:
        summary = result_df.copy().groupby(["Dataset", "Approach", "Parameters"]).mean().reset_index()
        summary = filter_best(summary, worst=False, median=True)
    sort_by = ["Dims", "Dataset", "Approach", "Parameters"]
    result_df = result_df.sort_values(by=sort_by)
    result_df = result_df.apply(func=add_params_to_df, axis=1)
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD (ae)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "ABCD (pca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "ABCD (kpca)"
    # result_df[result_df["Dataset"] == "Average"] = 0

    if print_summary:
        summary = summary.round(decimals={
            "F1": 2, "F0.5": 2, "F2": 2, "Prec.": 2, "Rec.": 2, "MTPO [ms]": 3, "MTD": 1
        })
        summary = summary.sort_values(by=sort_by)
        summary.drop(["Parameters", "Dims", "MTPO [ms]", "F0.5", "F2"], axis=1, inplace=True)
        print(summary.set_index(["Dataset", "Approach"]).to_latex(escape=False))
    abcd = np.logical_or(result_df["Approach"] == "ABCD (ae)",
                          result_df["Approach"] == "ABCD (pca)")
    abcd = np.logical_or(abcd, result_df["Approach"] == "ABCD (kpca)")
    abcd = result_df[abcd]
    abcd[r"$E$"] = abcd["E"].astype(int)
    abcd["Approach"][abcd["Approach"] == "ABCD (ae)"] = "ae"
    abcd["Approach"][abcd["Approach"] == "ABCD (kpca)"] = "kpca"
    abcd["Approach"][abcd["Approach"] == "ABCD (pca)"] = "pca"
    abcd = abcd.groupby(["Approach", "Parameters", "Dataset"]).mean().reset_index()
    abcd_eta = abcd.copy()
    average = abcd_eta.copy()  # .groupby(["Approach", r"$\eta$"]).mean().reset_index()
    average["Dataset"] = "Average"
    abcd_eta = abcd.sort_values(by="Dims")
    abcd_eta = pd.concat([average, abcd_eta], axis=0)
    abcd[r"$E$"] = abcd[r"$E$"].astype(int)

    if "RW (kpca)" in np.unique(result_df["Approach"]):
        avg_df = result_df.groupby(["Approach", "Dataset"]).mean().reset_index()
        avg_df["Window"] = ""
        avg_df["Model"] = ""
        for index, row in avg_df.iterrows():
            approach = row["Approach"]
            alg, model = approach.split(" ")
            model = model[1:-1]
            row["Window"] = "AW" if alg == "ABCD" else alg
            row["Model"] = model.upper()
            avg_df.loc[index] = row
        avg_df = avg_df[["Model", "Window", "F1", "Prec.", "Rec.", "MTD"]].sort_values(["Model", "Window"]).groupby(["Model", "Window"]).median()
        print(avg_df.round(decimals={
            "F1": 2, "Prec.": 2, "Rec.": 2, "MTD": 1
        }).to_latex(escape=False))
        del avg_df

    if plot_eta_sensitivity_study:
        g = sns.catplot(x=r"$\eta$", y="F1", col="Dataset", hue="Approach", errwidth=1, col_wrap=4,
                        data=abcd_eta, kind="bar", palette=sns.color_palette("Dark2"),
                        height=1 * 8 / 6, aspect=.6 * 5 / 4)
        axes = plt.gcf().axes
        plt.gcf().subplots_adjust(left=0.08)
        for i, ax in enumerate(axes):
            # ax.set_ylim(0.4, 1.0)
            # ax.set_ylim(bottom=.2)
            if i < 8:
                current_title = ax.get_title()
                dataset = current_title.split(" = ")[-1]
                ax.set_title(dataset)
            else:
                ax.set_title("")
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        plt.tight_layout(pad=0.5)
        plt.subplots_adjust(right=0.79)
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", "sensitivity-study-eta.pdf"))
        plt.show()

    if plot_E_sensitivity_study:
        abcd = abcd[abcd["Approach"] == "ae"]
        abcd_E = abcd.copy()
        average = abcd_E.copy() # '.groupby(["Approach", "Parameters", r"$E$"]).mean().reset_index()
        average["Dataset"] = "Average"
        abcd_E = abcd.sort_values(by="Dims")
        abcd_E = pd.concat([average, abcd_E], axis=0)
        n_colors = len(np.unique(abcd_E["Approach"]))
        ax = sns.barplot(x="Dataset", y="F1", hue="E", data=abcd_E, palette=sns.cubehelix_palette(n_colors=n_colors), errwidth=1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel("")
        ax.set_ylim(bottom=.3)
        ax.legend(title=r"$E$", loc='center left', bbox_to_anchor=(1, 0.5))
        change_bar_width(ax, .23)
        plt.gcf().set_size_inches((4.5, 1.8))
        plt.tight_layout()
        plt.subplots_adjust(right=0.79)
        plt.savefig(os.path.join(os.getcwd(), "..", "figures", "sensitivity-study-E.pdf"))
        plt.show()

    if plot_gradual_changes_comparison:
        result_df = result_df.groupby(["Approach", "Parameters", "Dataset"]).mean().reset_index()
        mean = result_df.groupby(["Approach", "Parameters"]).mean().reset_index()
        mean["Dataset"] = "Average"
        result_df = pd.concat([mean, result_df])
        result_df["MTD (thousands)"] = result_df["MTD"] / 1000
        value_vars = ["F1", "Prec.", "Rec.", "MTD (thousands)"]
        melted_df = pd.melt(result_df, id_vars=["Dataset", "Approach"], value_vars=value_vars,
                            var_name="Metric", value_name="Value")
        melted_df = melted_df[melted_df["Value"].isna() == False]
        g = sns.catplot(data=melted_df, x="Approach", y="Value", col="Dataset", row="Metric", kind="box",
                        linewidth=0.7, fliersize=2, sharey="row", palette=sns.color_palette("Dark2")
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
        plt.gcf().set_size_inches(3.5 * 2.5, 3.5 * 2)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(os.path.join("..", "figures", "evaluation_gradual_changes.pdf"))
        plt.show()

        g = sns.catplot(data=melted_df[melted_df["Approach"] != "Average"],
                        x="Approach", y="Value", row="Metric", kind="box",
                        linewidth=0.7, fliersize=2, sharey="row", palette=sns.color_palette("Dark2")
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
        plt.xticks(rotation=45, ha='right')
        plt.gcf().set_size_inches(3.8, 4.7)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        plt.savefig(os.path.join("..", "figures", "evaluation_gradual_changes_presentation.pdf"))
        plt.show()


def add_mean_column(df: pd.DataFrame):
    mean_df = df.groupby(["Approach"]).mean().reset_index()
    mean_df["Dataset"] = "Average"
    mean_df["Parameters"] = ""
    df = pd.concat([df, mean_df], ignore_index=True).sort_values(by=["Dataset", "Approach", "Parameters"])
    return df.set_index(["Dataset", "Approach", "Parameters"])


def filter_best(df, worst: bool, median: bool, add_mean: bool = True):
    df = df.sort_values(by=["F1", "MTD"])
    median_indices = []
    min_indices = []
    max_indices = []
    indices = []
    df = df.groupby(["Dataset", "Approach", "Parameters"]).mean().reset_index()
    for _, gdf in df.groupby(["Dataset", "Approach"]):
        if np.all(gdf["F1"].isna()):
            max_index = gdf.index[0]
            indices.append(max_index)
            continue
        gdf = gdf.dropna()
        if len(gdf) == 0:
            continue
        max_index = gdf["F1"].idxmax()
        indices.append(max_index)
    #     if "ABCD0 (ae)" in gdf["Approach"].to_numpy():
    #         max_indices.append(max_index)
    #         if median:
    #             med = gdf["F1"].dropna().median()
    #             median_index = (gdf["F1"] - med).abs().idxmin()
    #             if median_index == max_index and len(gdf) > 1:  # second clause should be particularly relevant for testing.
    #                 median_index = (gdf["F1"] - med).abs().drop(max_index).idxmin()
    #             median_indices.append(median_index)
    #         if worst:
    #             min_index = gdf["F1"].idxmin()
    #             min_indices.append(min_index)
    # if median:
    #     indices += median_indices
    #     df["Approach"].loc[median_indices] = "ABCD (med)"
    # if worst:
    #     indices += min_indices
    #     df["Approach"].loc[min_indices] = "ABCD (min)"
    df["Approach"].loc[max_indices] = "ABCD (max)"
    # indices = np.unique(indices)
    # df = df.loc[indices]
    if add_mean:
        df = add_mean_column(df)
    return df.reset_index()


if __name__ == '__main__':
    compare(print_summary=True,
            plot_gradual_changes_comparison=True)
