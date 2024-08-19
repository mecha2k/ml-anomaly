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
from sklearn.preprocessing import MinMaxScaler

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

load_from_file = True


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


def plot_target_correlation(
    df: pd.DataFrame,
    target_col: str = "target",
    n_top_features: int = 30,
    color_sequence: list[str] | None = None,
    template_theme: str = "plotly_white",
) -> None:

    correlations = df.corr()[target_col]
    correlations_abs = correlations.abs().sort_values(ascending=False)
    top_correlations = correlations[correlations_abs.index[1 : n_top_features + 1]]

    feature_names = top_correlations.index
    correlation_values = top_correlations.values

    # Set up color scale
    if color_sequence is None:
        color_sequence = [
            "#0d0887",
            "#46039f",
            "#7201a8",
            "#9c179e",
            "#bd3786",
            "#d8576b",
            "#ed7953",
            "#fb9f3a",
            "#fdca26",
            "#f0f921",
        ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=feature_names,
            x=correlation_values,
            orientation="h",
            marker=dict(
                color=correlation_values,
                colorscale=color_sequence,
                colorbar=dict(title="Correlation"),
            ),
        )
    )
    fig.update_layout(
        title=f"<b>Top {n_top_features} Features Correlated with {target_col.capitalize()}</b>",
        xaxis_title="<b>Correlation Coefficient</b>",
        yaxis_title="<b>Feature</b>",
        height=800,
        width=1200,
        template=template_theme,
    )
    # Add vertical line at x=0 for reference
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.5,
        x1=0,
        y1=len(feature_names) - 0.5,
        line=dict(color="black", width=1, dash="dash"),
    )

    fig.savefig(Path("../saved/images") / "correlation.png")


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scaler = MinMaxScaler().fit(df)
    output = scaler.transform(df)
    output = pd.DataFrame(output, columns=train_df.columns, index=list(train_df.index.values))

    x = np.array(output, dtype=np.float32)
    over_1 = np.any(x > 1.0)
    below_0 = np.any(x < 0.0)
    is_nan = np.any(np.isnan(x))
    print(f"Any data over 1.0: {over_1}, below 0: {below_0}, none: {is_nan}")
    return df


if __name__ == "__main__":
    seed_everything()

    data_path = Path("../datasets/HAICon2021")
    if load_from_file:
        train_df_raw = pd.read_pickle(data_path / "train/train_df_raw.pkl")
        valid_df_raw = pd.read_pickle(data_path / "validation/valid_df_raw.pkl")
        test_df_raw = pd.read_pickle(data_path / "test/test_df_raw.pkl")
    else:
        train_file, valid_file, test_file = prepare_datafiles(data_path)

        train_df_raw = dataframe_from_csvs(train_file)
        valid_df_raw = dataframe_from_csvs(valid_file)
        test_df_raw = dataframe_from_csvs(test_file)
        train_df_raw.to_pickle(data_path / "train/train_df_raw.pkl")
        valid_df_raw.to_pickle(data_path / "validation/valid_df_raw.pkl")
        test_df_raw.to_pickle(data_path / "test/test_df_raw.pkl")
    print(train_df_raw.shape, valid_df_raw.shape, test_df_raw.shape)

    train_df_raw = train_df_raw[:10000]
    target_col = train_df_raw.columns.drop(["timestamp"])
    train_df = train_df_raw[target_col].astype(float)
    train_df = normalize_dataframe(train_df)
    plot_target_correlation(train_df, target_col=target_col, n_top_features=30)
    print(train_df.describe())

    # print(fitted.data_max_)
    # ## 출력 결과
    # ## [891.       1.       3.      80.       8.       6.     512.3292]
    # output = min_max_scaler.transform(output)
    # output = pd.DataFrame(output, columns=df.columns, index=list(df.index.values))
    # print(output.head())
    #
    # plot_target_correlation(train_df, target_col=target_col, n_top_features=30)

    # extract independent variables from correlation analysis and generate heatmap graph
    # corr = train_df_raw.corr()
    # corr = corr.abs()
    # corr = corr[corr > 0.9]
    # corr = corr[corr < 1]
    # corr = corr.dropna(axis=0, how="all")
    # corr = corr.dropna(axis=1, how="all")
    # corr = corr.fillna(0)
    # plt.figure(figsize=(12, 12))
    # plt.matshow(corr, fignum=1)
    # plt.colorbar()
    # plt.show()
