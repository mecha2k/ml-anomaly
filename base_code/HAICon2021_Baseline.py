import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dateutil
import sys
import pickle

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import timedelta
from tqdm.notebook import trange
from TaPR_pkg import etapr


INIT_FLAG = True

MAKE_DATAFRAME = INIT_FLAG
MAKE_TRAIN_DATASET = INIT_FLAG
MAKE_VALID_DATASET = INIT_FLAG
MAKE_TEST_DATASET = INIT_FLAG
TRAIN = INIT_FLAG

TRAIN_DATASET = sorted([x for x in Path("../datasets/HAICon2021/train").glob("*.csv")])
TEST_DATASET = sorted([x for x in Path("../datasets/HAICon2021/test").glob("*.csv")])
VALIDATION_DATASET = sorted(
    [x for x in Path("../datasets/HAICon2021/validation").glob("*.csv")]
)


TIMESTAMP_FIELD = "timestamp"
IDSTAMP_FIELD = "id"
ATTACK_FIELD = "attack"


def dataframe_from_csv(target):
    return pd.read_csv(target).rename(columns=lambda x: x.strip())


def dataframe_from_csvs(targets):
    return pd.concat([dataframe_from_csv(x) for x in targets])


def normalize(df):
    ndf = df.copy()
    for c in df.columns:
        if TAG_MIN[c] == TAG_MAX[c]:
            ndf[c] = df[c] - TAG_MIN[c]
        else:
            ndf[c] = (df[c] - TAG_MIN[c]) / (TAG_MAX[c] - TAG_MIN[c])
    return ndf


# boundary_check 함수는 Pandas Dataframe에 있는 값 중 1 초과의 값이 있는지, 0 미만의 값이 있는지, NaN이 있는지 점검합니다.
def boundary_check(df):
    x = np.array(df, dtype=np.float32)
    return np.any(x > 1.0), np.any(x < 0), np.any(np.isnan(x))


TRAIN_DF_RAW = dataframe_from_csvs(TRAIN_DATASET)

# 학습 데이터셋은 공격을 받지 않은 평상시 데이터이고 시간을 나타내는 필드인 timestamp가 있으며, 나머지 필드는 모두 비식별화된
# 센서/액추에이터의 값입니다.정규화는 센서/액추에이터 필드만을 대상으로 해야 합니다.
# 본 문서에서는 전체 데이터를 대상으로 이상을 탐지하므로 "attack" 필드만 사용하였습니다.
# VALID_COLUMNS_IN_TRAIN_DATASET은 학습 데이터셋에 있는 모든 센서/액추에이터 필드를 담고 있습니다.
# 가끔 학습 데이터셋에 존재하지 않는 필드가 테스트 데이터셋에 존재하는 경우가 있습니다.
# 학습 시 보지 못했던 필드에 대해서 테스트를 할 수 없으므로 학습 데이터셋을 기준으로 필드 이름을 얻어냈습니다.
VALID_COLUMNS_IN_TRAIN_DATASET = TRAIN_DF_RAW.columns.drop([TIMESTAMP_FIELD])

TAG_MIN = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].min()
TAG_MAX = TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET].max()


def make_train_df():
    # TRAIN_DF는 정규화를 마친 후 exponential weighted function을 통과시킨 결과입니다.
    # 센서에서 발생하는 noise를 smoothing 시켜주기를 기대하고 적용했습니다.
    TRAIN_DF = (
        normalize(TRAIN_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
    )
    print(TRAIN_DF.describe())
    # 1보다 큰 값, 0보다 작은 값, not a number가 없습니다. 정규화가 정상적으로 처리되었습니다.
    print(boundary_check(TRAIN_DF))

    return TRAIN_DF, TRAIN_DF_RAW, VALID_COLUMNS_IN_TRAIN_DATASET


train_path = Path("../datasets/HAICon2021/train")
if not MAKE_DATAFRAME:
    TRAIN_DF = pd.read_pickle(train_path / "train_df.pkl")
    TRAIN_DF_RAW = pd.read_pickle(train_path / "train_df_raw.pkl")
else:
    TRAIN_DF, TRAIN_DF_RAW, _ = make_train_df()
    TRAIN_DF.to_pickle(train_path / "train_df.pkl")
    TRAIN_DF_RAW.to_pickle(train_path / "train_df_raw.pkl")


# ## 학습 모델 설정 & 데이터 입출력 정의
# 딥러닝 학습과 추론에는 PyTorch를 사용했습니다.
# 베이스라인 모델은 Stacked RNN(GRU cells)을 이용해서 이상을 탐지합니다.
# 정상 데이터만 학습해야 하고, 정상 데이터에는 어떠한 label도 없으므로 unsupervised learning을 해야 합니다.
#
# 본 모델에서는 슬라이딩 윈도우를 통해 시계열 데이터의 일부를 가져와서 해당 윈도우의 패턴을 기억하도록 했습니다.
# 슬라이딩 윈도우는 90초(HAI는 1초마다 샘플링되어 있습니다)로 설정했습니다.
#
# 모델의 입출력은 다음과 같이 설정했습니다.
# - 입력 : 윈도우의 앞부분 89초에 해당하는 값
# - 출력 : 윈도우의 가장 마지막 초(90번째 초)의 값
#
# 이후 탐지 시에는 모델이 출력하는 값(예측값)과 실제로 들어온 값의 차를 보고 차이가 크면 이상으로 간주했습니다.
# 많은 오차가 발생한다는 것은 기존에 학습 데이터셋에서 본 적이 없는 패턴이기 때문이라는 가정입니다.


WINDOW_GIVEN = 89
WINDOW_SIZE = 90


class HaiDataset(Dataset):
    def __init__(self, timestamps, df, stride=1, attacks=None):
        self.ts = np.array(timestamps)
        self.tag_values = np.array(df, dtype=np.float32)
        self.valid_idxs = []
        for L in trange(len(self.ts) - WINDOW_SIZE + 1):
            R = L + WINDOW_SIZE - 1
            if dateutil.parser.parse(self.ts[R]) - dateutil.parser.parse(
                self.ts[L]
            ) == timedelta(seconds=WINDOW_SIZE - 1):
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
        last = i + WINDOW_SIZE - 1
        item = {"attack": self.attacks[last]} if self.with_attack else {}
        item["ts"] = self.ts[i + WINDOW_SIZE - 1]
        item["given"] = torch.from_numpy(self.tag_values[i : i + WINDOW_GIVEN])
        item["answer"] = torch.from_numpy(self.tag_values[last])
        return item


# HaiDataset 클래스는 PyTorch의 Dataset 인터페이스를 정의한 것입니다.
# 데이터셋을 읽을 때는 슬라이딩 윈도우가 유효한 지 점검합니다.
# 정상적인 윈도우라면 원도우의 첫 시각과 마지막 시각의 차가 89초가 되어야 합니다.
# stride 파라미터는 슬라이딩을 할 때 크기를 의미합니다.
# 전체 윈도우를 모두 학습할 수도 있지만, 시계열 데이터에서는 슬라이딩 윈도우를 1초씩 적용하면 이전 윈도우와 다음 윈도우의 값이 거의 같습니다.
# 본 노트북에서는 학습을 빠르게 마치기 위해 10초씩 건너뛰면서 데이터를 추출하도록 했습니다.
# (물론 슬라이딩 크기를 1로 설정하여 모든 데이터셋을 보게 하면 더 좋을 것입니다)


dataset_path = Path("../datasets/HAICon2021/train") / "dataset_train.pkl"
if MAKE_TRAIN_DATASET:
    HAI_DATASET_TRAIN = HaiDataset(TRAIN_DF_RAW[TIMESTAMP_FIELD], TRAIN_DF, stride=10)
    with open(dataset_path, "wb") as f:
        pickle.dump(HAI_DATASET_TRAIN, f)
    print(HAI_DATASET_TRAIN[0])
else:
    with open(dataset_path, "rb") as f:
        HAI_DATASET_TRAIN = pickle.load(f)
    print(f"train dataset loaded : {len(HAI_DATASET_TRAIN)}")


# 데이터셋이 잘 로드되는 것을 볼 수 있습니다.
# 모델은 3층 bidirectional GRU를 사용합니다.
# Hidden cell의 크기는 100으로 설정했습니다.
# Dropout은 사용하지 않았습니다.
# 모델이 윈도우의 가장 첫 번째 값과 RNN의 출력을 더해서 내보내도록 skip connection(forward 메소드의 return 문 참조)을 만들었습니다.

N_HIDDENS = 100
N_LAYERS = 3
BATCH_SIZE = 512


class StackedGRU(torch.nn.Module):
    def __init__(self, n_tags):
        super().__init__()
        self.rnn = torch.nn.GRU(
            input_size=n_tags,
            hidden_size=N_HIDDENS,
            num_layers=N_LAYERS,
            bidirectional=True,
            dropout=0,
        )
        self.fc = torch.nn.Linear(N_HIDDENS * 2, n_tags)

    def forward(self, x):
        x = x.transpose(0, 1)  # (batch, seq, params) -> (seq, batch, params)
        self.rnn.flatten_parameters()
        outs, _ = self.rnn(x)
        out = self.fc(outs[-1])
        return x[0] + out


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"{device} is available in torch")

MODEL = StackedGRU(n_tags=TRAIN_DF.shape[1])
MODEL.to(device=device)


# ## 신규 모델 학습
# 모델 학습을 직접 하려면 아래 코드를 실행하시면 됩니다.
# 이미 학습된 모델을 로드해서 결과만 보시려면 아래 '모델 불러오기' section으로 가셔서 실행을 이어가시면 됩니다.
# Loss function은 MSE를 선택했고, optimizer는 AdamW(Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019)를 사용합니다.
# 학습 시 epoch loss가 가장 좋았던 모델의 파라미터를 저장합니다.
# 학습은 32 에포크 진행했습니다.
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
            given = batch["given"].to(device=device)
            guess = model(given)
            answer = batch["answer"].to(device=device)
            loss = loss_fn(answer, guess)
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


model_path = Path("../saved/model.pt")
image_path = Path("../saved/images")

if TRAIN:
    MODEL.train()
    BEST_MODEL, LOSS_HISTORY = train(HAI_DATASET_TRAIN, MODEL, BATCH_SIZE, 32)
    print(BEST_MODEL["loss"], BEST_MODEL["epoch"])

    with open(model_path, "wb") as f:
        model_dict = {
            "state": BEST_MODEL["state"],
            "best_epoch": BEST_MODEL["epoch"],
            "loss_history": LOSS_HISTORY,
        }
        torch.save(model_dict, f)

## 모델 불러오기
with open(model_path, "rb") as f:
    SAVED_MODEL = torch.load(f)

MODEL.load_state_dict(SAVED_MODEL["state"])

plt.figure(figsize=(16, 4))
plt.title("Training Loss Graph")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.yscale("log")
plt.plot(SAVED_MODEL["loss_history"])
plt.savefig(image_path / "loss.png")

# ## 학습된 모델을 이용한 탐지
# 검증 데이터셋을 불러와서 모델에 입력으로 주고 예측값과 실제값의 차를 얻어봅니다.
# 검증 데이터셋에 대해서도 만들어둔 함수를 이용해서 점검해봅니다.
# Not a number가 있는지 점검하는 것이 주요 목적입니다.


def make_validation_df():
    VALIDATION_DF_RAW = dataframe_from_csvs(VALIDATION_DATASET)
    VALIDATION_DF = normalize(VALIDATION_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET])
    boundary_check(VALIDATION_DF)
    return VALIDATION_DF, VALIDATION_DF_RAW


valid_path = Path("../datasets/HAICon2021/validation")
if MAKE_DATAFRAME:
    VALIDATION_DF, VALIDATION_DF_RAW = make_validation_df()
    VALIDATION_DF.to_pickle(valid_path / "validation_df.pkl")
    VALIDATION_DF_RAW.to_pickle(valid_path / "validation_df_raw.pkl")
else:
    VALIDATION_DF = pd.read_pickle(valid_path / "validation_df.pkl")
    VALIDATION_DF_RAW = pd.read_pickle(valid_path / "validation_df_raw.pkl")


# 공격 데이터셋에서는 확실히 정상 데이터의 최솟값과 최댓값을 벗어나는 값이 나타나고 있습니다.
if MAKE_VALID_DATASET:
    HAI_DATASET_VALIDATION = HaiDataset(
        VALIDATION_DF_RAW[TIMESTAMP_FIELD],
        VALIDATION_DF,
        attacks=VALIDATION_DF_RAW[ATTACK_FIELD],
    )
    with open(valid_path / "dataset_validation.pkl", "wb") as f:
        pickle.dump(HAI_DATASET_VALIDATION, f)
    print(HAI_DATASET_VALIDATION[0])
else:
    with open(valid_path / "dataset_validation.pkl", "rb") as f:
        HAI_DATASET_VALIDATION = pickle.load(f)
    print(f"validation dataset loaded : {len(HAI_DATASET_VALIDATION)}")


# 검증 데이터셋에 대해서도 PyTorch Dataset 인스턴스를 만들었습니다.
# 모든 데이터 포인트에 대해 점검해야 하므로 학습 데이터 때와는 다르게 슬라이딩의 크기는 1로 두어야 합니다.
def inference(dataset, model, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    ts, dist, att = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            given = batch["given"].to(device=device)
            answer = batch["answer"].to(device=device)
            guess = model(given)
            ts.append(np.array(batch["ts"]))
            dist.append(torch.abs(answer - guess).cpu().numpy())
            try:
                att.append(np.array(batch["attack"]))
            except KeyError:
                att.append(np.zeros(batch_size))

    return (
        np.concatenate(ts),
        np.concatenate(dist),
        np.concatenate(att),
    )


# inference 함수는 데이터를 순차적으로 보면서 모델이 예측한 값과 실제 값의 차를 구해서 기록합니다.
MODEL.eval()
CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_VALIDATION, MODEL, BATCH_SIZE)

# CHECK_DIST는 검증 데이터셋 전체 시간대에 대해 모든 필드의 |예측값 - 실제값|을 가지고 있습니다.
print(CHECK_DIST.shape)

# 공격 여부 판단을 위해 같은 시각에서 전체 필드가 산출하는 차의 평균을 계산합니다.
ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)


# 결과를 눈으로 확인하기 위해 그래프를 그려보겠습니다.
# piece 파라미터는 그래프를 몇 개로 나누어 그릴지를 결정합니다. 세세한 결과를 보고 싶을 경우 숫자를 늘리면 됩니다.
def check_graph(xs, att, piece=2, THRESHOLD=None, name="default"):
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
        if THRESHOLD is not None:
            axs[i].axhline(y=THRESHOLD, color="r")
    plt.savefig(image_path / name)


THRESHOLD = 0.026
check_graph(
    ANOMALY_SCORE, CHECK_ATT, piece=5, THRESHOLD=THRESHOLD, name="anomaly_score"
)


# 주황색 선은 공격 위치를 나타내고, 파란색 선은 (평균) 오차의 크기를 나타냅니다.
# 전반적으로 공격 위치에서 큰 오차를 보이고 있습니다.
# 임의의 threshold(빨간색 선)가 넘어갈 경우 공격으로 간주합니다.
# 공격은 1로 정상은 0으로 표기합니다.
def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs


# 위의 그래프를 보면 대략 0.022를 기준으로 설정할 수 있을 것으로 보입니다.
# 여러 번의 실험을 통해 정밀하게 임계치를 선택하면 더 좋은 결과를 얻을 수 있을 것으로 예상합니다.
LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
print(LABELS, LABELS.shape)

# 정답지(ATTACK_LABELS)도 동일하게 추출합니다.
# 검증 데이터셋에 공격 여부를 나타내는 필드에는 정상을 0으로 공격을 1로 표기하고 있습니다.
# 위에 정의한 put_labels 함수를 이용해서 0.5를 기준으로 같은 방식으로 TaPR을 위한 label을 붙여줍니다.
ATTACK_LABELS = put_labels(np.array(VALIDATION_DF_RAW[ATTACK_FIELD]), threshold=0.5)
print(ATTACK_LABELS, ATTACK_LABELS.shape)


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


FINAL_LABELS = fill_blank(
    CHECK_TS, LABELS, np.array(VALIDATION_DF_RAW[TIMESTAMP_FIELD])
)
print(FINAL_LABELS.shape)

# ## 평가
# 평가는 TaPR을 사용합니다.
# 정답(ATTACK_LABELS)과 모델의 결과(FINAL_LABELS)의 길이가 같은지 확인합니다.
TaPR = etapr.evaluate_haicon(anomalies=ATTACK_LABELS, predictions=FINAL_LABELS)
print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")

# ## 테스트 데이터셋 예측
# 학습 데이터셋과 검증 데이터셋을 이용해 만든 모델로 테스트 데이터셋 결과를 예측합니다.
test_path = Path("../datasets/HAICon2021/test")
if MAKE_DATAFRAME:
    TEST_DF_RAW = dataframe_from_csvs(TEST_DATASET)
    TEST_DF = (
        normalize(TEST_DF_RAW[VALID_COLUMNS_IN_TRAIN_DATASET]).ewm(alpha=0.9).mean()
    )
    boundary_check(TEST_DF)
    TEST_DF.to_pickle(test_path / "test_df.pkl")
    TEST_DF_RAW.to_pickle(test_path / "test_df_raw.pkl")
else:
    TEST_DF = pd.read_pickle(test_path / "test_df.pkl")
    TEST_DF_RAW = pd.read_pickle(test_path / "test_df_raw.pkl")


if MAKE_TEST_DATASET:
    HAI_DATASET_TEST = HaiDataset(TEST_DF_RAW[TIMESTAMP_FIELD], TEST_DF, attacks=None)
    with open(test_path / "dataset_test.pkl", "wb") as f:
        pickle.dump(HAI_DATASET_TEST, f)
    print(HAI_DATASET_TEST[0])
else:
    with open(test_path / "dataset_test.pkl", "rb") as f:
        HAI_DATASET_TEST = pickle.load(f)
    print(f"test dataset loaded : {len(HAI_DATASET_TEST)}")


# 테스트 데이터셋에 대해서도 PyTorch Dataset 인스턴스를 만들었습니다. 모든 데이터 포인트에 대해 점검해야 하므로 검증 데이터 때와
# 마찬가지로 슬라이딩의 크기는 1로 두어야 합니다.
# inference 함수로 데이터를 순차적으로 보면서 모델이 예측한 값과 실제 값의 차를 구해서 기록합니다.
MODEL.eval()
CHECK_TS, CHECK_DIST, CHECK_ATT = inference(HAI_DATASET_TEST, MODEL, BATCH_SIZE)

# 공격 여부 판단을 위해 같은 시각에서 전체 필드가 산출하는 차의 평균을 계산합니다.
ANOMALY_SCORE = np.mean(CHECK_DIST, axis=1)

# 결과를 눈으로 확인하기 위해 그래프를 그려보겠습니다.
check_graph(
    ANOMALY_SCORE, CHECK_ATT, piece=5, THRESHOLD=THRESHOLD, name="anomaly_score_test"
)

# 검증 데이터셋을 이용해 찾은 threshold를 이용해 공격 여부를 예측합니다.
LABELS = put_labels(ANOMALY_SCORE, THRESHOLD)
print(LABELS, LABELS.shape)

# 예측한 결과를 제출양식에 맞춰 저장합니다.
submission_path = Path("../datasets/HAICon2021")
submission = pd.read_csv(submission_path / "sample_submission.csv")
submission.index = submission["timestamp"]
submission.loc[CHECK_TS, "attack"] = LABELS
submission.to_csv(submission_path / "baseline.csv", index=False)
# **Copyright 2020. ETRI부설국가보안기술연구소. All rights reserved.**
