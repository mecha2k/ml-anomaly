import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import random
import datetime

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path


epochs = 1
batch_size = 1024
learning_rate = 1e-5
stride = 10
window_size = 41
window_given = 40

n_layers = 3
n_hiddens = 150
n_hiddens_2 = 70

load_from_pickle = True


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"{device} is available in torch")


seed_everything()
data_path = Path("../datasets/open")

if load_from_pickle:
    train_df_raw = pd.read_pickle(data_path / "train_raw.pkl")
    test_df_raw = pd.read_pickle(data_path / "test_raw.pkl")
else:
    train_df_raw = pd.read_csv(data_path / "train.csv")
    train_df_raw.to_pickle(data_path / "train_raw.pkl")
    test_df_raw = pd.read_csv(data_path / "test.csv")
    test_df_raw.to_pickle(data_path / "test_raw.pkl")


def normalize_dataframe(train_df, test_df):
    scaler = MinMaxScaler().fit(train_df)
    train_ls = scaler.transform(train_df)
    test_ls = scaler.transform(test_df)
    train_df = pd.DataFrame(train_ls, columns=train_df.columns, index=list(train_df.index.values))
    test_df = pd.DataFrame(test_ls, columns=test_df.columns, index=list(test_df.index.values))

    x = np.array(train_df, dtype=np.float32)
    over_1 = np.any(x > 1.0)
    below_0 = np.any(x < 0.0)
    is_nan = np.any(np.isnan(x))
    print(f"Any data over 1.0: {over_1}, below 0: {below_0}, none: {is_nan}")
    return train_df, test_df


columns_target = train_df_raw.columns.drop(["Timestamp", "anomaly"])
train_df = train_df_raw[columns_target].astype(float)
test_df = test_df_raw[columns_target].astype(float)
train_df, test_df = normalize_dataframe(train_df, test_df)
print(train_df.describe())
print(test_df.describe())


class TimeSeriesDataset(Dataset):
    def __init__(self, ts, df, stride=1, window_size=41, window_given=40):
        self.ts = np.array(ts)
        self.vals = np.array(df, dtype=np.float32)

        self.valid_idx = np.arange(0, len(self.ts) - window_size + 1, stride)
        self.num_win = len(self.valid_idx)

        self.pre_ts = self.ts[self.valid_idx + window_size - 1]
        self.pre_in = np.array([self.vals[i : i + window_given] for i in self.valid_idx])
        self.pre_tgt = self.vals[self.valid_idx + window_size - 1]

    def __len__(self):
        return self.num_win

    def __getitem__(self, idx):
        return {
            "timestamps": self.pre_ts[idx],
            "input": torch.from_numpy(self.pre_in[idx]),
            "target": torch.from_numpy(self.pre_tgt[idx]),
        }


train_dataset = TimeSeriesDataset(
    train_df_raw["Timestamp"],
    train_df,
    stride=stride,
    window_size=window_size,
    window_given=window_given,
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataset), len(train_loader))


class GRU_Linear(nn.Module):
    def __init__(self, n_tags, n_hiddens=150, n_hiddens_2=70, n_layers=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_tags,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(n_hiddens * 2, n_hiddens_2)
        self.dense = nn.Linear(n_hiddens_2, n_tags)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_sequence):
        input_sequence = input_sequence.transpose(0, 1)
        self.gru.flatten_parameters()
        gru_outputs, _ = self.gru(input_sequence)
        last_gru_output = gru_outputs[-1]

        output = self.fc(last_gru_output)
        output = self.relu(output)
        output = self.dense(output)
        output = torch.sigmoid(output)

        return output


def train_model(model, train_loader, optimizer, criterion, n_epochs, device):
    train_losses = []
    best_model = {"loss": float("inf"), "state": None, "epoch": 0}

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", unit="batch") as t:
            for batch in t:
                inputs = batch["input"].to(device)
                targets = batch["target"].to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                t.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}, Average Train Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_model["loss"]:
            best_model["state"] = model.state_dict()
            best_model["loss"] = avg_epoch_loss
            best_model["epoch"] = epoch + 1

    return train_losses, best_model


model = GRU_Linear(
    n_tags=train_df.shape[1], n_hiddens=n_hiddens, n_hiddens_2=n_hiddens_2, n_layers=n_layers
).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses, best_model = train_model(
    model, train_loader, optimizer, criterion, epochs, device=device
)

model.eval()
train_errors = []
with torch.no_grad():
    for batch in train_loader:
        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        outputs = model(inputs)
        errors = torch.mean(torch.abs(targets - outputs), dim=1).cpu().numpy()
        train_errors.extend(errors)

threshold = np.mean(train_errors) + 2 * np.std(train_errors)


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


test_dataset = TimeSeriesDataset(
    test_df_raw["Timestamp"],
    test_df,
    stride=stride,
    window_size=window_size,
    window_given=window_given,
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
timestamps, distances = inference(model, test_loader, device=device)
anomaly_score = np.mean(distances, axis=1)

labels = put_labels(anomaly_score, threshold)
prediction = fill_blank(timestamps, labels, np.array(test_df_raw["Timestamp"]))
prediction = prediction.flatten().tolist()

sample_submission = pd.read_csv(data_path / "sample_submission.csv")
sample_submission["anomaly"] = prediction
sample_submission.to_csv(data_path / "final_submission.csv", encoding="UTF-8-sig", index=False)
print(sample_submission["anomaly"].value_counts())
