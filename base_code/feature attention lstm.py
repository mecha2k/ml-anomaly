import torch
import numpy as np
import pandas as pd
import dateutil
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import trange
from TaPR_pkg import etapr
from pathlib import Path
from datetime import timedelta
from scipy import signal

import random
import pickle
import sys
import os

epochs = 2
batch_size = 2048

stride = 40
window_size = 40
window_given = 39
threshold = 0.026

load_from_file = False

LEAV_IDX = [
    11,
    15,
    22,
    23,
    26,
    27,
    29,
    30,
    31,
    39,
    42,
    52,
    55,
    58,
    61,
    64,
    67,
    69,
    72,
    74,
    75,
    76,
    80,
    82,
]


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


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


def prepare_datafiles(paths):
    train = sorted([x for x in (Path(paths) / "train").glob("*.csv")])
    valid = sorted([x for x in (Path(paths) / "validation").glob("*.csv")])
    test = sorted([x for x in (Path(paths) / "test").glob("*.csv")])
    return train, valid, test


def normalize(df, tag_min, tag_max):
    ndf = df.copy()
    for c in df.columns:
        if tag_min[c] == tag_max[c]:
            ndf[c] = df[c] - tag_min[c]
        else:
            ndf[c] = (df[c] - tag_min[c]) / (tag_max[c] - tag_min[c])
    return ndf


def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return f"Any data over 1.0 : {np.any(x > 1.0)}, below 0 : {np.any(x < 0)}, none : {np.any(np.isnan(x))}"


class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=1, attacks=None, window_size=40, window_given=39):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        self.window_size = window_size
        self.window_given = window_given
        for L in trange(len(self.ts) - self.window_size + 1):
            R = L + self.window_size - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(self.ts[L]) == timedelta(
                seconds=self.window_size - 1
            ):
                self.valid_idxs.append(L)
        self.valid_idxs = np.array(self.valid_idxs, dtype=np.int32)[::stride]
        self.n_idxs = len(self.valid_idxs)
        print(f"# of valid windows: {self.n_idxs}")
        if attacks is not None:
            self.attacks = np.array(attacks, dtype=np.float32)
            self.with_attack = True
        else:
            self.with_attack = False

    def __len__(self):
        return self.n_idxs

    def __getitem__(self, idx):
        i = self.valid_idxs[idx]
        last = i + self.window_size - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + self.window_size - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + self.window_given])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item


class StackedLSTM(torch.nn.Module):
    def __init__(self, n_hiddens, n_layers):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=24,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = torch.nn.Linear(n_hiddens * 2, 24)
        self.relu = torch.nn.LeakyReLU(0.1)

        # mix up을 적용하기 위해서 learnable parameter인 w를 설정합니다.
        w = torch.nn.Parameter(torch.FloatTensor([-0.01]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w

        self.sigmoid = torch.nn.Sigmoid()

        # feature attention을 위한 dense layer를 설정합니다.
        self.dense1 = torch.nn.Linear(24, 12)
        self.dense2 = torch.nn.Linear(12, 24)

    def forward(self, x):
        x = x[:, :, LEAV_IDX]  # batch, window_size, params

        pool = torch.nn.AdaptiveAvgPool1d(1)

        attention_x = x
        attention_x = attention_x.transpose(1, 2)  # batch, params, window_size

        attention = pool(attention_x)  # batch, params, 1

        connection = attention  # 이전 정보를 저장하고 있습니다.
        connection = connection.reshape(-1, 24)  # batch, params

        # feature attention을 적용합니다.
        attention = self.relu(torch.squeeze(attention))
        attention = self.relu(self.dense1(attention))
        attention = self.sigmoid(
            self.dense2(attention)
        )  # sigmoid를 통해서 (batch, params)의 크기로 확률값이 나타나 있는 attention을 생성합니다.

        x = x.transpose(0, 1)  # (batch, window_size, params) -> (window_size, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(self.relu(outs[-1]))  # 이전 대회 코드를 보고 leaky relu를 추가했습니다.

        mix_factor = self.sigmoid(self.w)  # w의 값을 비율로 만들어 주기 위해서 sigmoid를 적용합니다.

        return mix_factor * connection * attention + out * (1 - mix_factor)  # 이전 정보


def train(dataset, model, batch_size, n_epochs):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = torch.nn.MSELoss()
    epochs = trange(n_epochs, desc="training")
    best = {"loss": sys.float_info.max}
    loss_history = []
    for e in epochs:
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            given = batch["given"].cuda()
            answer = batch["answer"].cuda()
            guess = model(given)
            loss = loss_fn(
                answer[:, LEAV_IDX], guess
            )  # answer도 86개의 feature가 있기 때문에 LEAV_IDX만 사용하기 위해 작성
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        loss_history.append(epoch_loss)
        epochs.set_postfix_str(f"loss: {epoch_loss:.6f}")
        if epoch_loss < best["loss"]:
            best["state"] = model.state_dict()
            best["loss"] = epoch_loss
            best["epoch"] = e + 1
    return best, loss_history


def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].to(device=device)
            answer = batch["answer"].to(device=device)
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer[:, LEAV_IDX] - guess).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except KeyError:
                att.append(np.zeros(batch_size))

    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )


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
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if threshold is not None:
            axs[i].axhline(y=threshold, color="r")
    plt.savefig(Path("../saved/images") / name)


# 주황색 선은 공격 위치를 나타내고, 파란색 선은 (평균) 오차의 크기를 나타냅니다.
# 전반적으로 공격 위치에서 큰 오차를 보이고 있습니다.
# 임의의 threshold(빨간색 선)가 넘어갈 경우 공격으로 간주합니다.
# 공격은 1로 정상은 0으로 표기합니다.
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs


# 탐지 모델이 윈도우 방식으로 판단을 진행했기 때문에,
# 1. 첫 시작의 몇 초는 판단을 내릴 수 없고
# 2. 데이터셋 중간에 시간이 연속되지 않는 구간에 대해서는 판단을 내릴 수 없습니다.
# 위에서 보시는 바와 같이 정답에 비해 얻어낸 label의 수가 적습니다.
# 아래의 fill_blank 함수는 빈칸을 채워줍니다.
# 빈 곳은 정상(0) 표기하고 나머지는 모델의 판단(정상 0, 비정상 1)을 채워줍니다.
def fill_blank(check_ts, labels, total_ts):
    def ts_generator():
        for t in total_ts:
            yield dateutil.parser.parse(t)

    def label_generator():
        for t, label in zip(check_ts, labels):
            yield dateutil.parser.parse(t), label

    g_ts = ts_generator()
    g_label = label_generator()
    final_labels = []

    try:
        current = next(g_ts)
        ts_label, label = next(g_label)
        while True:
            if current > ts_label:
                ts_label, label = next(g_label)
                continue
            elif current < ts_label:
                final_labels.append(0)
                current = next(g_ts)
                continue
            final_labels.append(label)
            current = next(g_ts)
            ts_label, label = next(g_label)
    except StopIteration:
        return np.array(final_labels, dtype=np.int8)


if __name__ == "__main__":
    seed_everything()

    data_path = Path("../datasets/HAICon2021")
    train_file, valid_file, test_file = prepare_datafiles(data_path)

    train_df_raw = dataframe_from_csvs(train_file)
    valid_df_raw = dataframe_from_csvs(valid_file)
    test_df_raw = dataframe_from_csvs(test_file)
    print(train_df_raw.describe())

    timestamp = "timestamp"
    idstamp = "id"
    attack = "attack"

    valid_columns = train_df_raw.columns.drop([timestamp])
    tag_min = train_df_raw[valid_columns].min()
    tag_max = train_df_raw[valid_columns].max()

    if load_from_file:
        train_df = pd.read_pickle(data_path / "train/train_df.pkl")
        valid_df = pd.read_pickle(data_path / "validation/valid_df.pkl")
        test_df = pd.read_pickle(data_path / "test/test_df.pkl")

        with open(data_path / "train/train_ds.pkl", "rb") as f:
            hai_train_ds = pickle.load(f)
        with open(data_path / "validation/valid_ds.pkl", "rb") as f:
            hai_valid_ds = pickle.load(f)
        with open(data_path / "test/test_ds.pkl", "rb") as f:
            hai_test_ds = pickle.load(f)

    else:
        train_df = normalize(train_df_raw[valid_columns], tag_min, tag_max).ewm(alpha=0.9).mean()
        valid_df = normalize(valid_df_raw[valid_columns], tag_min, tag_max).ewm(alpha=0.9).mean()
        test_df = normalize(test_df_raw[valid_columns], tag_min, tag_max).ewm(alpha=0.9).mean()
        train_df.to_pickle(data_path / "train/train_df.pkl")
        valid_df.to_pickle(data_path / "validation/valid_df.pkl")
        test_df.to_pickle(data_path / "test/test_df.pkl")
        print(boundary_check(train_df))
        print(boundary_check(valid_df))
        print(boundary_check(test_df))

        hai_train_ds = HaiDataset(
            train_df_raw[timestamp],
            train_df,
            stride=stride,
            attacks=None,
            window_size=window_size,
            window_given=window_given,
        )
        hai_valid_ds = HaiDataset(
            valid_df_raw[timestamp],
            valid_df,
            stride=stride,
            attacks=valid_df_raw[attack],
            window_size=window_size,
            window_given=window_given,
        )
        hai_test_ds = HaiDataset(
            test_df_raw[timestamp],
            test_df,
            stride=stride,
            attacks=None,
            window_size=window_size,
            window_given=window_given,
        )

        with open(data_path / "train/train_ds.pkl", "wb") as f:
            pickle.dump(hai_train_ds, f)
        with open(data_path / "validation/valid_ds.pkl", "wb") as f:
            pickle.dump(hai_valid_ds, f)
        with open(data_path / "test/test_ds.pkl", "wb") as f:
            pickle.dump(hai_test_ds, f)

        print(hai_train_ds[0])
        print(hai_valid_ds[0])
        print(hai_test_ds[0])

    model = StackedLSTM(n_hiddens=200, n_layers=3)
    model.to(device=device)
    print(model)

    model.train()
    best_model, loss_history = train(hai_train_ds, model, batch_size, epochs)
    print(f"best model loss : {best_model['loss']:.5f}, epoch : {best_model['epoch']}")

    model_path = Path("../saved/model.pt")
    image_path = Path("../saved/images")

    with open(model_path, "wb") as f:
        model_dict = {
            "state": best_model["state"],
            "best_epoch": best_model["epoch"],
            "loss_history": loss_history,
        }
        torch.save(model_dict, f)

    with open(model_path, "rb") as f:
        saved_model = torch.load(f)

    model.load_state_dict(saved_model["state"])

    plt.figure(figsize=(16, 4))
    plt.title("Training Loss Graph")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.yscale("log")
    plt.plot(saved_model["loss_history"])
    plt.savefig(image_path / "loss.png")

    model.eval()
    check_ts, check_dist, check_att = inference(hai_valid_ds, model, batch_size)
    anomaly_score = np.mean(check_dist, axis=1)
    print(check_dist.shape)

    # 이전 대회의 데이터 후처리를 참고했습니다. moving average보다 lowpass filter를 적용하는 것이 성능이 좋아서 이를 선택했습니다.
    # b, a = signal.butter(N=1, Wn=0.02, btype="lowpass")
    # xs = signal.filtfilt(b, a, anomaly_score)
    check_graph(anomaly_score, check_att, piece=2, threshold=threshold, name="valid_anomaly")

    # 위의 그래프를 보면 대략 0.022를 기준으로 설정할 수 있을 것으로 보입니다.
    # 여러 번의 실험을 통해 정밀하게 임계치를 선택하면 더 좋은 결과를 얻을 수 있을 것으로 예상합니다.
    labels = put_labels(anomaly_score, threshold=threshold)
    print(labels.shape)

    # 정답지(ATTACK_LABELS)도 동일하게 추출합니다.
    # 검증 데이터셋에 공격 여부를 나타내는 필드에는 정상을 0으로 공격을 1로 표기하고 있습니다.
    # 위에 정의한 put_labels 함수를 이용해서 0.5를 기준으로 같은 방식으로 TaPR을 위한 label을 붙여줍니다.
    attack_labels = put_labels(np.array(valid_df_raw[attack]), threshold=0.5)
    print(attack_labels.shape)

    final_labels = fill_blank(check_ts, labels, np.array(valid_df_raw[timestamp]))
    print(final_labels.shape)

    # 정답(ATTACK_LABELS)과 모델의 결과(FINAL_LABELS)의 길이가 같은지 확인합니다.
    TaPR = etapr.evaluate_haicon(anomalies=attack_labels, predictions=final_labels)
    print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
    print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

    model.eval()
    check_ts, check_dist, check_att = inference(hai_test_ds, model, batch_size)
    anomaly_score = np.mean(check_dist, axis=1)
    check_graph(anomaly_score, check_att, piece=2, threshold=threshold, name="test_anomaly")

    labels = put_labels(anomaly_score, threshold=threshold)
    print(labels, labels.shape)

    submission = pd.read_csv(data_path / "sample_submission.csv")
    submission.index = submission["timestamp"]
    submission.loc[check_ts, "attack"] = labels
    submission.to_csv(data_path / "submission_final.csv", index=False)
    print(submission["attack"].value_counts())
