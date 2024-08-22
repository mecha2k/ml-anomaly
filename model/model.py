import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import torch


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class GRU_Linear(BaseModel):
    def __init__(self, input_size=100, n_hiddens=150, n_hiddens_2=70, n_layers=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(n_hiddens * 2, n_hiddens_2)
        self.dense = nn.Linear(n_hiddens_2, input_size)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, input_sequence):
        input_sequence = input_sequence.transpose(0, 1)
        self.gru.flatten_parameters()
        gru_outputs, _ = self.gru(input_sequence)
        last_gru_output = gru_outputs[-1]
        output = self.fc(last_gru_output)
        output = self.relu(output)
        output = self.dense(output)
        return torch.sigmoid(output)


class StackedLSTM(BaseModel):
    def __init__(self, input_size, n_hiddens, n_layers):
        super().__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=n_hiddens,
            num_layers=n_layers,
            bidirectional=True,
            dropout=0.1,
        )
        self.fc = torch.nn.Linear(n_hiddens * 2, input_size)
        self.relu = torch.nn.LeakyReLU(0.1)

        # mix up을 적용하기 위해서 learnable parameter인 w를 설정합니다.
        w = torch.nn.Parameter(torch.FloatTensor([-0.01]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w

        self.sigmoid = torch.nn.Sigmoid()

        # feature attention을 위한 dense layer를 설정합니다.
        self.dense1 = torch.nn.Linear(input_size, input_size // 2)
        self.dense2 = torch.nn.Linear(input_size // 2, input_size)

    def forward(self, x):
        # x = x[:, :, LEAV_IDX]  # batch, window_size, params

        pool = torch.nn.AdaptiveAvgPool1d(1)

        attention_x = x
        attention_x = attention_x.transpose(1, 2)  # batch, params, window_size

        attention = pool(attention_x)  # batch, params, 1

        connection = attention  # 이전 정보를 저장하고 있습니다.
        connection = connection.reshape(-1, x.shape[-1])  # batch, params

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

        mix_factor = self.sigmoid(
            self.w
        )  # w의 값을 비율로 만들어 주기 위해서 sigmoid를 적용합니다.

        return mix_factor * connection * attention + out * (1 - mix_factor)  # 이전 정보
