from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import torch
import pickle


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
        num_workers=1,
        window_size=41,
        window_given=40,
        stride=10,
        init_loader=False,
        training=True,
    ):
        self.data_dir = Path(data_dir)
        self._init_dataloader(init_loader, stride, window_size, window_given)
        self.dataset = self.train_dataset if training else self.test_dataset

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

    @staticmethod
    def _check_dataframe(df, mode="train"):
        x = np.array(df, dtype=np.float32)
        over_1 = np.any(x > 1.0)
        below_0 = np.any(x < 0.0)
        is_nan = np.any(np.isnan(x))
        print(f"Any {mode} data over 1.0: {over_1}, below 0: {below_0}, none: {is_nan}")

    def _init_dataloader(self, init_loader, stride, window_size, window_given):
        if not init_loader:
            self.train_df_raw = pd.read_pickle(self.data_dir / "train_raw.pkl")
            self.train_df = pd.read_pickle(self.data_dir / "train.pkl")
            self.test_df_raw = pd.read_pickle(self.data_dir / "test_raw.pkl")
            self.test_df = pd.read_pickle(self.data_dir / "test.pkl")
            with open(self.data_dir / "train_dataset.pkl", "rb") as f:
                self.train_dataset = pickle.load(f)
            with open(self.data_dir / "test_dataset.pkl", "rb") as f:
                self.test_dataset = pickle.load(f)

        else:
            self.train_df_raw = pd.read_csv(self.data_dir / "train.csv")
            self.train_df_raw.to_pickle(self.data_dir / "train_raw.pkl")

            columns_target = self.train_df_raw.columns.drop(["Timestamp", "anomaly"])
            train_df = self.train_df_raw[columns_target].astype(float)
            scaler = MinMaxScaler().fit(train_df)
            train_ls = scaler.transform(train_df)
            self.train_df = pd.DataFrame(
                train_ls, columns=train_df.columns, index=list(train_df.index.values)
            )
            self.train_df.to_pickle(self.data_dir / "train.pkl")

            self.test_df_raw = pd.read_csv(self.data_dir / "test.csv")
            self.test_df_raw.to_pickle(self.data_dir / "test_raw.pkl")

            test_df = self.test_df_raw[columns_target].astype(float)
            test_ls = scaler.transform(test_df)
            self.test_df = pd.DataFrame(
                test_ls, columns=test_df.columns, index=list(test_df.index.values)
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
            with open(self.data_dir / "train_dataset.pkl", "wb") as f:
                pickle.dump(self.train_dataset, f)
            with open(self.data_dir / "test_dataset.pkl", "wb") as f:
                pickle.dump(self.test_dataset, f)
