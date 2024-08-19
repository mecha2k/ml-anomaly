from torchvision import datasets, transforms
from torch.utils.data import Dataset
from base import BaseDataLoader


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
    ):

        self.data_dir = data_dir
        self.df_raw = pd.read_pickle(data_dir / "train.pkl")
        self.dataset = TimeSeriesDataset(
            train_df_raw["Timestamp"],
            train_df,
            stride=stride,
            window_size=window_size,
            window_given=window_given,
        )
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
