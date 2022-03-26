import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from changeds.metrics import true_positives, false_positives, false_negatives, recall, precision, fb_score, \
    mean_until_detection, jaccard, mean_cp_detection_time_error
from scipy.stats import spearmanr
from tqdm import tqdm
import seaborn as sns

from E_images import ename
from util import get_last_experiment_dir, str_to_arr, fill_df, get_abcd_hyperparameters_from_str


import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{nicefrac}'
mpl.rc('font', family='serif')


if __name__ == '__main__':
    last_exp_dir = get_last_experiment_dir(ename)
    all_files = os.listdir(last_exp_dir)
    result = {}

    result_df = []
    for file in tqdm(all_files):
        df = pd.read_csv(os.path.join(last_exp_dir, file), sep=",", index_col=0).convert_dtypes()
        df = fill_df(df)
        approach = np.unique(df["approach"])[0]
        dataset = np.unique(df["dataset"])[0]
        changes = [i for i in df.index if df["change-point"].loc[i]]
        relevant_info = df.loc[changes]
        ground_truth = relevant_info["dims-gt"].iloc[0]
        found = relevant_info["dims-found"].iloc[0]
        ground_truth = str_to_arr(ground_truth, dtype=int)
        found = str_to_arr(found, dtype=int)
        result[dataset] = [ground_truth, found]
    for dataset in result:
        data = result[dataset]
        print(data)
        image = np.zeros(28*28)
        image[data] = 1
        sns.heatmap(result[dataset.reshape((28, 28))])
        plt.show()
