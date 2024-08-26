import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from pathlib import Path
from datetime import datetime


def check_graphs_v1(data, preds, threshold=None, name="default", piece=15):
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


def check_graphs_v2(data, preds, anomaly, interval=10000, img_path=None, mode="train"):
    pieces = int(len(data) // interval)
    for i in range(pieces):
        start = i * interval
        end = min(start + interval, len(data))
        xticks = list(range(start, end, 1000))
        values = data[start:end]
        plt.figure(figsize=(16, 8))
        plt.ylim(-0.5, 1.5)
        plt.xticks(xticks)
        plt.grid()
        plt.plot(values)
        plt.plot(preds[start:end], color="green", linewidth=8)
        plt.plot(anomaly[start:end], color="blue", linewidth=5)
        plt.savefig(img_path / f"{mode}_raw_data" / f"raw_{i+1:02d}_pages")
        plt.close("all")


def fill_blank_data(timestamps, datasets, total_ts):
    # create dataframes with total_ts index and 0 values
    df_total = pd.DataFrame(0, index=total_ts, columns=["outputs"]).astype(float)
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

    threshold = np.percentile(anomaly_score, 95)
    anomaly_score = fill_blank_data(timestamps, anomaly_score, np.array(timestamps_raw))
    prediction = np.zeros_like(anomaly_score)
    prediction[anomaly_score > threshold] = 1
    check_graphs_v1(anomaly_score, prediction, threshold, name=image_path / "test_anomaly")

    # train_df = pd.read_pickle(data_path / "train.pkl")
    # train = train_df.values
    # check_graphs_v2(train, np.zeros_like(train), img_path=image_path, mode="train")
    test_df = pd.read_pickle(data_path / "test.pkl")
    check_graphs_v2(test_df.values, prediction, anomaly_score, img_path=image_path, mode="test")

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())
