import numpy as np
import pandas as pd
import torch
import datetime
from tqdm import tqdm


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

    labels = put_labels(anomaly_score, threshold)
    prediction = fill_blank(timestamps, labels, np.array(data_loader.test_df_raw["Timestamp"]))
    prediction = prediction.flatten().tolist()

    sample_submission = pd.read_csv(data_path / "sample_submission.csv")
    sample_submission["anomaly"] = prediction
    sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
    print(sample_submission["anomaly"].value_counts())
