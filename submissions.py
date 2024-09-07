import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import datetime
import pickle
from scipy import stats
from tqdm import tqdm
from pathlib import Path


def check_graph(xs, att, piece=2, threshold=None, name="default"):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].set_ylim(0, 0.3)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak)
        if threshold is not None:
            axs[i].axhline(y=threshold, color="r")
    plt.tight_layout()
    plt.savefig(name)


def inference(model, data_loader, device="cuda"):
    model.eval()
    timestamps = []
    distances = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Inference", unit="batch"):
            inputs = batch["input"].to(device)
            targets = batch["target"].to(device)

            predictions = model(inputs)

            timestamps.extend(batch["timestamps"])
            distances.extend(torch.abs(targets - predictions).cpu().tolist())

    return np.array(timestamps), np.array(distances)


def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs


def fill_blank(check_ts, labels, total_ts):
    TS_FORMAT = "%Y-%m-%d %H:%M:%S"

    def parse_ts(ts):
        return datetime.datetime.strptime(ts.strip(), TS_FORMAT)

    def ts_label_iter():
        return ((parse_ts(ts), label) for ts, label in zip(check_ts, labels))

    final_labels = []
    label_iter = ts_label_iter()
    cur_ts, cur_label = next(label_iter, (None, None))

    for ts in total_ts:
        cur_time = parse_ts(ts)
        while cur_ts and cur_time > cur_ts:
            cur_ts, cur_label = next(label_iter, (None, None))

        if cur_ts == cur_time:
            final_labels.append(cur_label)
            cur_ts, cur_label = next(label_iter, (None, None))
        else:
            final_labels.append(0)

    return np.array(final_labels, dtype=np.int8)


def get_threshold(anomaly_score, percentile):
    anomaly_score = anomaly_score[anomaly_score < 1]
    threshold = np.percentile(anomaly_score, percentile)
    return threshold


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
    df_total = pd.DataFrame(0, index=total_ts, columns=["outputs"]).astype(float)
    df_total.index = pd.to_datetime(df_total.index)
    df_partial = pd.DataFrame(datasets, index=timestamps, columns=["outputs"])
    df_partial.index = pd.to_datetime(df_partial.index)
    df_total.update(df_partial)
    return df_total["outputs"].values


def final_submission(model, data_loader, device, data_path):
    timestamps, distances = inference(model, data_loader, device=device)
    anomaly_score = np.mean(distances, axis=1)
    attacks = np.zeros_like(anomaly_score)
    timestamps_raw = data_loader.test_df_raw["Timestamp"]

    with open(data_path / "test_anomaly.pkl", "wb") as f:
        data_dict = {
            "timestamps": timestamps,
            "anomaly_score": anomaly_score,
            "attacks": attacks,
            "timestamps_raw": timestamps_raw,
        }
        pickle.dump(data_dict, f)

    image_path = Path("saved/images")
    # threshold = np.percentile(anomaly_score, 99)
    threshold = np.mean(anomaly_score) + 2 * np.std(anomaly_score)
    print(f"mean-std based Threshold: {threshold}")
    anomaly_score = fill_blank_data(timestamps, anomaly_score, np.array(timestamps_raw))
    prediction = np.zeros_like(anomaly_score)
    prediction[anomaly_score > threshold] = 1
    check_graphs(anomaly_score, prediction, threshold=threshold, name=image_path / "test_anomaly")

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())


if __name__ == "__main__":
    data_path = Path("datasets/open")
    image_path = Path("saved/images")

    with open(data_path / "test_anomaly.pkl", "rb") as f:
        data_dict = pickle.load(f)

    attacks = data_dict["attacks"]
    timestamps = data_dict["timestamps"]
    timestamps_raw = data_dict["timestamps_raw"]
    anomaly_score = data_dict["anomaly_score"]

    threshold = np.mean(anomaly_score) + 2 * np.std(anomaly_score)
    print(f"mean-std based Threshold: {threshold}")
    anomaly_score = fill_blank_data(timestamps, anomaly_score, np.array(timestamps_raw))
    prediction = np.zeros_like(anomaly_score)
    prediction[anomaly_score > threshold] = 1
    check_graphs(anomaly_score, prediction, threshold=threshold, name=image_path / "test_anomaly")

    # labels = put_labels(anomaly_score, threshold)
    # prediction = fill_blank(timestamps, labels, np.array(timestamps_raw))
    # prediction = prediction.flatten().tolist()
    #
    # sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    # sample_submission["anomaly"] = prediction
    # sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    # print(sample_submission["anomaly"].value_counts())
    #
    # amin = np.percentile(anomaly_score, 5)
    # amax = np.percentile(anomaly_score, 95)
    # plt.hist(anomaly_score, bins=100, density=True, range=(amin, amax + 0.1))
    # plt.legend(["Anomaly score"])
    # plt.grid()
    # plt.savefig("saved/images/anomaly_hist")
    #
    # print(f"Number of total data points: {len(anomaly_score)}")
    # print(f"threshold of anomaly score: ", threshold)
    #
    # train_df = pd.read_pickle(data_path / "train.pkl")
    # train_df_corr = train_df.corr()
    # for col in train_df.columns:
    #     print("========================================")
    #     print(train_df_corr[col][0.7 < train_df_corr[col]])
    #     print("========================================")
    #
    drop_cols = ["B_2","B_4","B_1","B_3","A_2","F_1","D_1","D_2","C_5","E_1","E_2","E_4","C_2"]  # fmt: skip
    # print(tuple(drop_cols))
    # train_df = train_df.drop(columns=drop_cols)

    # cols_name = ["A", "B", "C", "D", "E", "F"]
    # for col in cols_name:
    #     columns = [c for c in train_df.columns if c.startswith(col)]
    #     correlations = train_df[columns].corr()
    #     plt.figure(figsize=(12, 12))
    #     plt.title(f"Correlation Heatmap {col}")
    #     plt.imshow(correlations, cmap="coolwarm", interpolation="nearest")
    #     plt.colorbar()
    #     plt.xticks(range(len(columns)), columns, rotation=90)
    #     plt.yticks(range(len(columns)), columns)
    #     plt.savefig(f"saved/images/corr_{col}_col")

    # cols_name = ["A", "B", "C", "D", "E", "F"]
    # for col in cols_name:
    #     columns = [c for c in train_df.columns if c.startswith(col)]
    #     plt.figure(figsize=(24, 12))
    #     plt.plot(train_df[columns])
    #     plt.title(f"Train Data {col}")
    #     plt.xlabel("Index")
    #     plt.ylabel("Value")
    #     plt.grid()
    #     plt.legend()
    #     plt.savefig(f"saved/images/train_{col}_col")

    # anomaly_sampled = np.random.choice(anomaly_score, size=500, replace=False)
    # kde = stats.gaussian_kde(anomaly_sampled)
    # density = kde(anomaly_score)
    # # 10% percentile of the density is the threshold
    # # data below 10% diff. between pred. and target are normal
    # threshold = np.percentile(density, 10)
    # outliers = anomaly_score[density < threshold]

    # plt.figure(figsize=(12, 6))
    # x_range = np.linspace(0, 1, 1000)
    # plt.plot(x_range, kde(x_range), label="KDE")
    # plt.scatter(anomaly_score, np.zeros_like(anomaly_score), alpha=0.5, s=100, label="Data points")
    # plt.scatter(outliers, np.zeros_like(outliers), color="red", s=30, label="Outliers")
    # plt.axhline(y=threshold, color="r", label="Density Based Threshold")
    # plt.title("Outlier Detection using KDE")
    # plt.xlabel("Value")
    # plt.ylabel("Density")
    # plt.grid()
    # plt.legend()
    # plt.savefig("saved/images/kde_outliers")
