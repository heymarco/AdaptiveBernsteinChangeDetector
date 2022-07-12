import os

import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
import arff


def parse_bioliq():
    data_path = os.path.join(os.getcwd(), "..", "PSE", "Bioliq_S-MAB_1wx20.csv")
    result_path = os.path.join(os.getcwd(), "..", "PSE", "datasets", "bioliq.csv")
    data = pd.read_csv(data_path)
    data.drop("Time", axis=1)
    data.to_csv(result_path, index=False)


def parse_ova_colon():
    data_path = os.path.join(os.getcwd(), "..", "PSE", "OVA_colon.arff")
    result_path = os.path.join(os.getcwd(), "..", "PSE", "datasets", "ova_colon.csv")
    df = pd.DataFrame(loadarff(data_path)[0])
    df.to_csv(result_path, index=False)


def parse_canada_climate_data():
    data_path = os.path.join(os.getcwd(), "..", "PSE", "canada_climate.arff")
    result_path = os.path.join(os.getcwd(), "..", "PSE", "datasets", "canada_climate.csv")
    df = arff.load(open(data_path, 'r'))
    cols = [item[0] for item in df["attributes"]]
    df = pd.DataFrame(data=df["data"], columns=cols)
    df.to_csv(result_path, index=False)


def parse_kdd99():
    data_path = os.path.join(os.getcwd(), "..", "PSE", "KDDCup99.arff")
    result_path = os.path.join(os.getcwd(), "..", "PSE", "datasets", "network_intrusions.csv")
    df = pd.DataFrame(loadarff(data_path)[0])
    df.to_csv(result_path, index=False)

if __name__ == '__main__':
    parse_kdd99()
