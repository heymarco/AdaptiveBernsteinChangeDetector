import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from util import new_dir_for_experiment_with_name

if __name__ == '__main__':
    ename = "synthetic_data"
    approaches = ["ABCD0 (ae)", "ABCD0 (pca)", "ABCD0 (kpca)", "D3", "IKS", "AdwinK", "IBDD", "WATCH"]
    datasets = ["HSphere"]
    folder_nr = 28
    from_dir = os.path.join(os.getcwd(), "results", "experiments", ename, str(folder_nr))
    to_dir = new_dir_for_experiment_with_name(ename)
    all_files = os.listdir(from_dir)

    for file in tqdm(all_files):
        if file.startswith("cache"):
            continue
        fp = os.path.join(from_dir, file)
        csv = pd.read_csv(fp)
        approach = csv.iloc[0]["approach"]
        dataset = csv.iloc[0]["dataset"]
        if approach in approaches:
            if not datasets:
                csv.to_csv(os.path.join(to_dir, file), index=False)
            else:
                if dataset not in datasets:
                    csv.to_csv(os.path.join(to_dir, file), index=False)
