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
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak)
        if threshold is not None:
            axs[i].axhline(y=threshold, color="r")
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


def final_submission(model, data_loader, threshold, device, data_path):
    timestamps, distances = inference(model, data_loader, device=device)
    anomaly_score = np.mean(distances, axis=1)
    attacks = np.zeros_like(anomaly_score)

    with open(data_path / "test_anomaly.pkl", "wb") as f:
        data_dict = {
            "timestamps": timestamps,
            "anomaly_score": anomaly_score,
            "attacks": attacks,
        }
        pickle.dump(data_dict, f)

    threshold = 0.268
    check_graph(
        anomaly_score, attacks, piece=2, threshold=threshold, name=data_path / "test_anomaly"
    )

    labels = put_labels(anomaly_score, threshold)
    prediction = fill_blank(timestamps, labels, np.array(data_loader.test_df_raw["Timestamp"]))
    prediction = prediction.flatten().tolist()

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())


if __name__ == "__main__":
    with open("datasets/open/test_anomaly.pkl", "rb") as f:
        data_dict = pickle.load(f)

    anomaly_score = data_dict["anomaly_score"]
    plt.hist(anomaly_score, bins=100, density=True, range=(0.26, 0.28))
    plt.legend(["Anomaly score"])
    plt.grid()
    plt.savefig("saved/images/anomaly_hist")

    # Generate sample data
    np.random.seed(42)
    data = np.concatenate(
        [np.random.normal(0, 1, 1000), np.random.normal(6, 1, 20)]  # Adding some outliers
    )

    # Compute KDE
    kde = stats.gaussian_kde(data)

    # Calculate the probability density for each point
    density = kde(data)

    # Define the threshold for outliers (e.g., bottom 1% of density)
    threshold = np.percentile(density, 5)

    # Identify outliers
    outliers = data[density < threshold]

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot the KDE
    x_range = np.linspace(data.min(), data.max(), 1000)
    plt.plot(x_range, kde(x_range), label="KDE")

    # Plot the data points
    plt.scatter(data, np.zeros_like(data), alpha=0.5, s=100, label="Data points")

    # Highlight outliers
    plt.scatter(outliers, np.zeros_like(outliers), color="red", s=30, label="Outliers")

    plt.title("Outlier Detection using KDE")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("saved/images/kde_outliers")

    print(f"Number of outliers detected: {len(outliers)}")
    print(f"Threshold : {threshold}")

    # # Generate 1D data with outliers (for demonstration)
    # np.random.seed(0)  # For reproducibility
    # data = np.concatenate(
    #     (
    #         np.random.normal(loc=5, scale=2, size=100),
    #         [50],  # An outlier at position 50
    #         np.random.normal(loc=10, scale=1.5, size=200),
    #     )
    # )
    #
    # # Sort the data for easier KDE calculation and plotting
    # data.sort()
    #
    # # Calculate KDE density
    # x = np.linspace(min(data), max(data), 1000)
    # kde = stats.gaussian_kde(data)
    # density = kde(x)
    #
    # # Plot KDE distribution with original data points overlaid
    # plt.figure(figsize=(10, 6))
    # plt.plot(x, density, label="KDE Density")
    # plt.scatter(
    #     data, [1] * len(data), c="red", alpha=0.5, label="Data Points"
    # )  # Use a placeholder value to make scatter visible
    # plt.xlabel("Value")
    # plt.ylabel("Density")
    # plt.legend()
    # plt.title("Kernel Density Estimate of Data with Outlier")
    # plt.savefig("saved/images/kernel_density")
    #
    # # Define and calculate the threshold based on density at data points (adjust as necessary)
    # threshold = np.percentile(density, 95)
    # # This might need adjustment based on actual dataset characteristics
    #
    # # Identify outliers by comparing to the calculated KDE density
    # outliers = [x for x, d in zip(data, density) if d < threshold]
    # print("Outliers:", outliers)
