import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

from E_visualize_loss import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, create_cache_dir_if_needed, get_E_and_eta, \
    get_abcd_hyperparameters_from_str, get_d3_hyperparameters_from_str, move_legend_below_graph, cm2inch

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')

if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    result_df = []
    j = 0
    for file in tqdm(all_files):
        j += 1
        # if j > 10:
        #     break
        df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",", index_col=0).convert_dtypes()
        df = fill_df(df)
        df["metric"] = df["metric"].rolling(30).mean()
        df["Num Observations"] = np.arange(len(df))
        result_df.append(df)
    result_df = pd.concat(result_df, ignore_index=True)
    result_df[r"$|\mathcal{W}|$"] = np.nan
    result_df[r"$\eta$"] = np.nan
    result_df["MTPO"] = np.nan
    result_df[r"$k_{max}$"] = ""
    result_df[r"$t$"] = np.nan
    result_df[r"$E$"] = 0
    for (rep, ndims, approach, params), rep_data in tqdm(
        result_df.groupby(["rep", "ndims", "approach", "parameters"])):
        E, eta = get_E_and_eta(params)
        n_splits = get_abcd_hyperparameters_from_str(params)[-1]
        result_df.loc[rep_data.index, r"$E$"] = E
        result_df.loc[rep_data.index, r"$\eta$"] = eta
    # result_df[r"$|W|$"] = result_df[r"$|\mathcal{W}|$"]
    # result_df[r"$d$"] = result_df["ndims"].astype(int)
    result_df["Approach"] = result_df["approach"]
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "ABCD (pca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "ABCD (ae)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "ABCD (kpca)"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (dummy)"] = "ABCD (id)"
    result_df = result_df.sort_values(by=["Approach", "dataset"])

    result_df = result_df[result_df[r"$\eta$"] == 0.3]

    grid = sns.relplot(
        data=result_df, x="Num Observations", y="metric",
        row="Approach", col="dataset", hue=r"$E$",
        kind="line",
        ci=None,
        facet_kws={"sharex": False, "sharey": False},
        zorder=1000
    )

    axes = grid.axes
    for row, (_, row_data) in zip(axes, result_df.groupby("Approach")):
        for ax, (_, ax_data) in zip(row, row_data.groupby("dataset")):
            for _, row in ax_data[np.logical_and(ax_data[r"$E$"] == 50, ax_data["is-change"])].iterrows():
                ax.axvline(row["Num Observations"], color="black", ls="dashed", lw=0.7, zorder=0)
            for _, E_data in ax_data.groupby(r"$E$"):
                palette = sns.cubehelix_palette(n_colors=81)
                E = E_data[r"$E$"].iloc[0]
                color = palette[E - 20]
                lims = ax.get_ylim()
                if np.any(E_data["in-pre-train"]):
                    ax.fill_between(E_data["Num Observations"], -1, 1, where=E_data["in-pre-train"],
                                    color="white", lw=0, zorder=100 - E)
                    ax.fill_between(E_data["Num Observations"], -1, 1, where=E_data["in-pre-train"],
                                    color=color, alpha=0.5, lw=0, zorder=101 - E)
                    ax.set_ylim(lims)
                else:
                    for _, row in E_data[E_data["change-point"]].iterrows():
                        ax.axvline(row["Num Observations"], color=color, alpha=0.5)

    plt.savefig(os.path.join(os.getcwd(), "..", "figures", ename + ".pdf"))
    plt.show()
