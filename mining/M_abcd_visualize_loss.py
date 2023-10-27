import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text
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
        df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",", index_col=0).convert_dtypes()
        df = fill_df(df)
        df = df[df["in-pre-train"] == False]
        df["metric"] = df["metric"][::-1].rolling(50).mean()[::-1]  # set the value at the left edge of the rolling window
        df["Stream length"] = np.arange(len(df))
        df = df[np.logical_and(df.index >= 1000, df.index < 5000)]
        result_df.append(df)
    result_df = pd.concat(result_df, ignore_index=True)
    result_df[r"$|\mathcal{W}|$"] = np.nan
    result_df[r"$\eta$"] = np.nan
    result_df["MTPO"] = np.nan
    result_df[r"$k_{max}$"] = ""
    result_df[r"$t$"] = np.nan
    result_df[r"$E$"] = np.nan
    result_df = result_df.rename({"metric": "MSE"}, axis=1)
    for (rep, ndims, approach, params), rep_data in tqdm(
        result_df.groupby(["rep", "ndims", "approach", "parameters"])):
        E, eta = get_E_and_eta(params)
        n_splits = get_abcd_hyperparameters_from_str(params)[-1]
        result_df.loc[rep_data.index, r"$E$"] = E
        result_df.loc[rep_data.index, r"$\eta$"] = eta
    result_df["Approach"] = result_df["approach"]
    result_df["Approach"][result_df["Approach"] == "ABCD0 (pca)"] = "PCA"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (ae)"] = "AE"
    result_df["Approach"][result_df["Approach"] == "ABCD0 (kpca)"] = "KPCA"
    is_ae = result_df["Approach"] == "AE"
    result_df.loc[is_ae, "Approach"] = result_df["Approach"][is_ae] + result_df[r"$E$"][is_ae].apply(lambda x: f", $E={x}$")
    result_df.loc[result_df["Approach"] == "KPCA", r"$E$"] = 102  # enforce ordering in plot by setting E to higher values
    result_df.loc[result_df["Approach"] == "KPCA", r"$E$"] = 101  # same
    result_df = result_df.sort_values(by=["dataset", r"$E$"])

    grid = sns.relplot(
        data=result_df, x="Stream length", y="MSE",
        col="Approach", row="dataset", hue=r"$\eta$",
        kind="line",
        ci=None,
        facet_kws={"sharex": "col", "sharey": False, "legend_out": True},
        zorder=100,
        lw=0.5,
    )

    axes = grid.axes
    for row_index, (row, (_, row_data)) in enumerate(zip(axes, result_df.groupby("dataset"))):
        for col_index, (ax, (_, ax_data)) in enumerate(zip(row, row_data.groupby("Approach"))):
            title = ax.get_title()
            left, right = title.split(" | ")
            dataset = left.split(" = ")[-1]
            approach = right.split(" = ")[-1]
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_xlabel("", labelpad=10, verticalalignment="bottom")
            if col_index == 0:
                ylabel_text = dataset
                ax.set_ylabel(ylabel_text)
            for _, row in ax_data[np.logical_and(ax_data[r"$\eta$"] == 0.3, ax_data["is-change"])].iterrows():
                if row_index == 0:
                    ax.set_title(approach, pad=16)
                else:
                    ax.set_title("")
                ax.axvline(row["Stream length"], color="black", ls="dashed", lw=0.7, zorder=0)
            # for _, E_data in ax_data.groupby(r"$E$"):
            #     palette = sns.cubehelix_palette(n_colors=81)
            #     E = E_data[r"$E$"].iloc[0]
            #     color = palette[E - 20]
            #     lims = ax.get_ylim()
                # if np.any(E_data["in-pre-train"]):
                #     ax.fill_between(E_data["Stream length"], -1, 1, where=E_data["in-pre-train"],
                #                     color="white", lw=0, zorder=100 - E)
                #     ax.fill_between(E_data["Stream length"], -1, 1, where=E_data["in-pre-train"],
                #                     color=color, alpha=0.5, lw=0, zorder=100 - E)
                #     ax.set_ylim(lims)
                # else:
                #     for _, row in E_data[E_data["change-point"]].iterrows():
                #         ax.axvline(row["Stream length"], color=color, lw=0.5)

    grid._legend.set_title("")
    for t in grid._legend.texts:
        t.set_text(f"$\eta = {t.get_text()}$")
    handles = grid._legend.legendHandles
    labels = grid._legend.texts
    handles.append(Line2D([0], [0], color="black", linewidth=0.7, linestyle="dashed"))
    labels.append(Text(text="Change point"))
    grid._legend.remove()
    grid.add_legend(legend_data={l.get_text(): handle
                                 for l, handle in zip(labels, handles)})
    sns.move_legend(grid, "upper center", ncol=4, title=None, frameon=False)
    plt.gcf().set_size_inches(6.29921, 7)
    plt.gcf().supxlabel("Stream length")
    plt.gcf().supylabel("MSE")
    plt.tight_layout(pad=.2)
    plt.gcf().subplots_adjust(top=0.89, right=0.97, wspace=0.35, hspace=0.44, bottom=0.08, left=0.12)
    plt.savefig(os.path.join(os.getcwd(), "..", "figures", ename + ".pdf"))
    plt.show()
