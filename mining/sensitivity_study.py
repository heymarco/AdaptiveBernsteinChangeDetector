import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from changeds.metrics import true_positives, false_positives, false_negatives, precision, recall, fb_score

from util import get_last_experiment_dir, move_legend_below_graph, get_E_and_eta
from E_sensitivity_study import ename


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{times}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')

if __name__ == '__main__':
    eta_str = r"$\eta$"
    data_dir = get_last_experiment_dir(ename)
    files = os.listdir(data_dir)
    dfs = []
    print("Loading datasets")
    for file in files:
        df = pd.read_csv(os.path.join(data_dir, file)).convert_dtypes().ffill()
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    result_df = []
    print("Processing data")
    for (rep, dataset), this_df in tqdm(data.groupby(["rep", "dataset"])):
        for params, gdf in this_df.groupby("parameters"):
            E, eta = get_E_and_eta(params)
            dims = gdf["ndims"].iloc[0]
            true_cps = [i for i in range(len(gdf)) if gdf["is-change"].iloc[i]]
            cp_distance = true_cps[0]
            reported_cps = [i for i in range(len(gdf)) if gdf["change-point"].iloc[i]]
            tp = true_positives(true_cps, reported_cps, cp_distance)
            fp = false_positives(true_cps, reported_cps, cp_distance)
            fn = false_negatives(true_cps, reported_cps, cp_distance)
            prec = precision(tp, fp, fn)
            rec = recall(tp, fp, fn)
            f1 = fb_score(true_cps, reported_cps, T=2000)
            result_df.append([E, eta, dataset, dims, f1])
    columns = ["E", eta_str, "Dataset", "Dims", "F1"]
    result_df = pd.DataFrame(result_df, columns=columns).sort_values(by="Dims")
    all_datasets = np.unique(result_df["Dataset"])
    n_subplots = len(all_datasets)
    n_cols = min(3, n_subplots)
    n_rows = int(max(1, np.ceil(n_cols / 3)))
    fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax_index, ax in enumerate(axes.flatten()):
        this_data = result_df[result_df["Dataset"] == all_datasets[ax_index]]
        ax.set_title(all_datasets[ax_index])
        sns.lineplot(data=this_data, x=eta_str, y="F1", hue="E", ax=axes[ax_index])
    plt.gcf().set_size_inches((4, 3))
    move_legend_below_graph(axes, ncol=5, title=r"$E$")
    plt.tight_layout()
    plt.show()
