import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from E_abcd_runtime_sensitivity_study import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, get_E_and_eta, \
    get_abcd_hyperparameters_from_str, get_d3_hyperparameters_from_str, move_legend_below_graph, cm2inch

import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


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
            result_df.append(df)
        result_df = pd.concat(result_df, ignore_index=True)
        result_df[r"$|\mathcal{W}|$"] = np.nan
        result_df[r"$\eta$"] = np.nan
        result_df["MTPO"] = np.nan
        result_df[r"$k_{max}$"] = ""
        result_df[r"$t$"] = np.nan
        for (rep, ndims, approach, params), rep_data in tqdm(result_df.groupby(["rep", "ndims", "approach", "parameters"])):
            if "ABCD" in approach:
               eta = get_E_and_eta(params)[1]
               n_splits = get_abcd_hyperparameters_from_str(params)[-2]
            else:
                eta = np.nan
                n_splits = np.nan
            reported_cps = [i for i in rep_data.index if rep_data["change-point"].loc[i]]
            if len(reported_cps) > 0:
                print("Number of changes is {}".format(len(reported_cps)))
                continue
            rep_data[r"$t$"] = range(len(rep_data))
            rep_data[r"$\eta$"] = eta
            rep_data["time"] = rep_data["time"] - rep_data["time"].iloc[0]
            rep_data[r"$|\mathcal{W}|$"] = np.arange(len(rep_data))
            rep_data["MTPO"] = rep_data["time"].diff()
            rep_data[r"$k_{max}$"] = "{}".format(n_splits) if "st = ed" in params else "all"
            result_df.loc[rep_data.index] = rep_data
        result_df.to_csv(os.path.join(cache_dir, "cached.csv"), index=False)
    result_df["time"] = result_df["time"] / 10E6  # milliseconds
    result_df["MTPO [ms]"] = result_df["MTPO"] / 10E6
    result_df[r"$|W|$"] = result_df[r"$|\mathcal{W}|$"]
    result_df = result_df[result_df["MTPO [ms]"] < 10]
    result_df[r"$d$"] = result_df["ndims"].astype(int)
    result_df["Approach"] = result_df["approach"]
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "ABCD (pca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD (ae)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "ABCD (kpca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (dummy)"] = "ABCD (dummy)"

    result_df = result_df.groupby(["Approach", "parameters", r"$d$", r"$\eta$", r"$|W|$", r"$k_{max}$"]).mean().reset_index()
    result_df = result_df.sort_values(by=["Approach", r"$d$", r"$|W|$"])
    result_df = result_df.groupby(["Approach", "parameters", r"$d$", r"$\eta$"]).rolling(30).mean().reset_index()
    result_df[r"$d$"] = result_df[r"$d$"].apply(lambda x: r"${}$".format(int(x)))

    mask = result_df["Approach"] == "ABCD (ae)"
    mask = np.logical_or(mask, result_df["Approach"] == "ABCD (pca)")
    mask = np.logical_or(mask, result_df["Approach"] == "ABCD (kpca)")
    mask = np.logical_and(mask, result_df[r"$d$"] != r"$10000$")
    result_df = result_df[mask]

    n_colors = len(np.unique(result_df[r"$d$"]))
    g = sns.relplot(data=result_df[result_df[r"$k_{max}$"] == 10][::10],
                    x=r"$|W|$", y="MTPO [ms]", ci=None, facet_kws={"sharey": False},
                    col=r"$\eta$",  # col=r"$\eta$",
                    hue=r"$d$", style="Approach", kind="line",
                    height=cm2inch(4.5)[0], aspect=1,
                    palette=sns.cubehelix_palette(n_colors=n_colors+1)[1:])
    # plt.yscale("log")
    axes = plt.gcf().axes
    plt.gcf().subplots_adjust(left=0.08)
    for i, ax in enumerate(axes):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.xscale("log")
    plt.tight_layout(pad=.5)
    plt.gcf().subplots_adjust(right=0.77, wspace=.3)  # top=0.95, bottom=0.1, left=0.1,
    plt.savefig(os.path.join("..", "figures", "runtime_eta.pdf"))
    plt.show()
    #
    g = sns.relplot(data=result_df[result_df[r"$\eta$"] == 0.5][::10], x=r"$|W|$", y="MTPO [ms]", ci=None, facet_kws={"sharey": False},
                    col=r"$k_{max}$",  # col=r"$\eta$",
                    hue=r"$d$", style="Approach", kind="line",
                    height=cm2inch(4.5)[0], aspect=1, palette=sns.cubehelix_palette(n_colors=n_colors+1)[1:])
    # plt.yscale("log")
    axes = plt.gcf().axes
    plt.gcf().subplots_adjust(left=0.08)
    for i, ax in enumerate(axes):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # plt.xscale("log")
    plt.tight_layout(pad=.5)
    plt.gcf().subplots_adjust(right=0.77, wspace=.3)  # top=0.95, bottom=0.1, left=0.1,
    plt.savefig(os.path.join("..", "figures", "runtime.pdf"))
    plt.show()




