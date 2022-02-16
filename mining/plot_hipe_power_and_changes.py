import os

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

from util import move_legend_below_graph

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), "..", "results", "result.csv")
    df = pd.read_csv(path)
    df["w"] = df["w1"] + df["w2"]
    date_and_time = pd.read_csv(os.path.join(os.getcwd(), "..", "src",
                                             "stream-datasets", "data", "hipe", "cache", "df.csv"))
    date_and_time = pd.to_datetime(date_and_time["SensorDateTime"], utc=True)
    changes = [i for i, is_change in enumerate(df["is-change"]) if is_change]
    detected_changes = [i for i, change in enumerate(df["change-point"].fillna(False)) if change]
    change_points = [i - df["delay"][i] for i in detected_changes]
    window_size = 100
    df = df.rolling(window=window_size).mean()
    df["SensorDateTime"] = date_and_time
    fig, axes = plt.subplots(2, 1)
    for line in date_and_time.iloc[changes]:
        for ax in axes:
            ax.axvline(line, c="black", alpha=0.3)
    for line in date_and_time.iloc[detected_changes]:
        for ax in axes:
            ax.axvline(line, c="green", alpha=0.3)
    for line in date_and_time.iloc[change_points]:
        for ax in axes:
            ax.axvline(line, c="red")
    sns.lineplot(x="SensorDateTime", y="loss", data=df, ax=axes[0])

    path_to_data = os.path.join(os.getcwd(), "..", "src", "stream-datasets", "data", "hipe")
    scaler = MinMaxScaler()
    files = [file for file in os.listdir(path_to_data) if file.endswith(".csv")]
    palette = sns.cubehelix_palette(n_colors=len(files))
    for i, filename in enumerate(files):
        p = os.path.join(path_to_data, filename)
        df = pd.read_csv(p)
        name = df["Machine"].loc[0]
        df["P_kW"] = scaler.fit_transform(df["P_kW"].to_numpy().reshape(-1, 1))
        df["SensorDateTime"] = pd.to_datetime(df["SensorDateTime"], utc=True)
        power = df["P_kW"]
        window_size = 100
        power = power.rolling(window=window_size).mean()
        sns.lineplot(x=df["SensorDateTime"], y=power, ax=axes[1], label=name, color=palette[i])
    move_legend_below_graph(axes, 3, "")
    fig.autofmt_xdate()
    axes[1].fmt_xdata = mdates.DateFormatter('%m-%d')
    axes[1].set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


