import numpy as np
import pandas as pd
import torch
import pickle

from sympy.core.random import shuffle
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

# drop_cols = ["B_2", "B_4", "B_1", "B_3", "A_2", "F_1", "D_1", "D_2", "C_5", "E_1", "E_2", "E_4", "C_2"]  # fmt: skip
drop_cols = []


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


class TimeSeriesDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        shuffle=True,
        validation_split=0.0,
        window_size=41,
        window_given=40,
        stride=10,
        training=True,
    ):
        self.data_dir = Path(data_dir)
        self._init_dataloader(stride, window_size, window_given)
        self.dataset = self.train_dataset if training else self.test_dataset

        super().__init__(self.dataset, batch_size, shuffle, validation_split)

    @staticmethod
    def _check_dataframe(df, mode="train"):
        x = np.array(df, dtype=np.float32)
        over_1 = np.any(x > 1.0)
        below_0 = np.any(x < 0.0)
        is_nan = np.any(np.isnan(x))
        print(f"Any {mode} data over 1.0: {over_1}, below 0: {below_0}, none: {is_nan}")

    def _init_dataloader(self, stride, window_size, window_given):
        self.train_df_raw = pd.read_csv(self.data_dir / "train.csv")
        columns_target = self.train_df_raw.columns.drop(["Timestamp", "anomaly"] + drop_cols)
        train_df = self.train_df_raw[columns_target].astype(float)

        scaler = MinMaxScaler().fit(train_df)
        train_ls = scaler.transform(train_df)
        self.train_df = (
            pd.DataFrame(train_ls, columns=train_df.columns, index=list(train_df.index.values))
            .ewm(alpha=0.9)
            .mean()
        )
        self.train_df.to_pickle(self.data_dir / "train.pkl")

        self.test_df_raw = pd.read_csv(self.data_dir / "test.csv")
        test_df = self.test_df_raw[columns_target].astype(float)
        test_ls = scaler.transform(test_df)
        self.test_df = (
            pd.DataFrame(test_ls, columns=test_df.columns, index=list(test_df.index.values))
            .ewm(alpha=0.9)
            .mean()
        )
        self.test_df.to_pickle(self.data_dir / "test.pkl")

        self._check_dataframe(self.train_df, mode="train")
        self._check_dataframe(self.test_df, mode="test")

        self.train_dataset = TimeSeriesDataset(
            self.train_df_raw["Timestamp"],
            self.train_df,
            stride=stride,
            window_size=window_size,
            window_given=window_given,
        )
        self.test_dataset = TimeSeriesDataset(
            self.test_df_raw["Timestamp"],
            self.test_df,
            stride=stride,
            window_size=window_size,
            window_given=window_given,
        )


class HMCDataset(Dataset):
    def __init__(self, data, win_size=100, step=1, mode="train"):
        self.data = data
        self.mode = mode
        self.win_size = win_size
        self.step = step

    def __len__(self):
        if self.mode == "train" or self.mode == "test":
            return (self.data.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.data.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        return np.float32(self.data[index : index + self.win_size])


class HMCDataLoader(BaseDataLoader):
    def __init__(
        self,
        data_dir,
        batch_size,
        win_size=100,
        training=True,
    ):
        data_path = Path(data_dir)
        self.scaler = MinMaxScaler()

        train_df_raw = pd.read_csv(data_path / "train.csv")
        train_df_raw = train_df_raw[:1000]
        data_columns = train_df_raw.columns.drop(["Timestamp", "anomaly"])
        train_df = train_df_raw[data_columns].astype(float)
        scaler = self.scaler.fit(train_df)
        train = scaler.transform(train_df)
        train_df = pd.DataFrame(train, columns=train_df.columns, index=list(train_df.index.values))
        train_df = train_df.ewm(alpha=0.9).mean()
        train_df.to_pickle(data_path / "train.pkl")

        test_df_raw = pd.read_csv(data_path / "test.csv")
        test_df_raw = test_df_raw[:1000]
        test_df = test_df_raw[data_columns].astype(float)
        test = scaler.transform(test_df)
        test_df = pd.DataFrame(test, columns=test_df.columns, index=list(test_df.index.values))
        test_df = test_df.ewm(alpha=0.9).mean()
        test_df.to_pickle(data_path / "test.pkl")

        self.train_df = train_df
        self.test_df = test_df
        self.test_timestamps = test_df_raw["Timestamp"]
        self.train = np.array(train_df.values)
        self.test = np.array(test_df.values)

        if training:
            shuffle = True
            self.dataset = HMCDataset(self.train, win_size, 1, "train")
        else:
            shuffle = False
            self.dataset = HMCDataset(self.test, win_size, 1, "test")

        super().__init__(self.dataset, batch_size, shuffle, validation_split=0.0)
