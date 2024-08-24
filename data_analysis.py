import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
from datetime import datetime
from submissions import put_labels


def check_graphs(data, preds, piece=15, threshold=None, name="default"):
    interval = len(data) // piece
    fig, axes = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = range(start, end)
        axes[i].set_ylim(0, 1)
        axes[i].plot(xticks, preds[start:end])
        axes[i].plot(xticks, data[start:end])
        if threshold is not None:
            axes[i].axhline(y=threshold, color="r")
    plt.tight_layout()
    plt.savefig(name)
    plt.close("all")


def fill_blank_data(timestamps, datasets, total_ts):
    # create dataframes with total_ts index and 0 values
    df_total = pd.DataFrame(0, index=total_ts, columns=["outputs"])
    df_total.index = pd.to_datetime(df_total.index)
    df_partial = pd.DataFrame(datasets, index=timestamps, columns=["outputs"])
    df_partial.index = pd.to_datetime(df_partial.index)
    df_total.update(df_partial)
    return df_total["outputs"].values


def anomaly_prediction(scores, piece=15):
    mean_std, percentile = [], []
    interval = len(scores) // piece
    for i in range(piece):
        start = i * interval
        end = min(start + interval, len(scores))
        mean_std.append(scores[start:end].mean() + 2 * scores[start:end].std())
        percentile.append(np.percentile(scores[start:end], 99))
    return mean_std, percentile


if __name__ == "__main__":
    data_path = Path("datasets/open")
    image_path = Path("saved/images")
    with open(data_path / "test_anomaly.pkl", "rb") as f:
        data_dict = pickle.load(f)

    timestamps = data_dict["timestamps"]
    timestamps_raw = data_dict["timestamps_raw"]
    anomaly_score = data_dict["anomaly_score"]

    threshold = np.percentile(anomaly_score, 99)
    anomaly_score = fill_blank_data(timestamps, anomaly_score, np.array(timestamps_raw))
    prediction = np.zeros_like(anomaly_score)
    prediction[anomaly_score > threshold] = 1
    mean_std, percentile = anomaly_prediction(anomaly_score, piece=15)
    check_graphs(anomaly_score, prediction, threshold=threshold, name=image_path / "test_anomaly")

    test_df = pd.read_pickle(data_path / "test.pkl")
    for columns in test_df.columns.values:
        check_graphs(test_df[columns].values, prediction, name=image_path / f"{columns}_test_preds")

    # sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    # sample_submission["anomaly"] = prediction
    # print(sample_submission["anomaly"].value_counts())
