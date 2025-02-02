# tcn_model.py
import torch
import torch.nn as nn
from base_model import BaseModel


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout_p
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight.data)
        nn.init.kaiming_normal_(self.conv2.weight.data)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight.data)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TCNModel(BaseModel):
    def __init__(self, x_dim, y_dim, num_channels, kernel_size, dropout_p, device):
        super().__init__(x_dim, y_dim, device)
        self.num_levels = len(num_channels)
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        self.build(num_channels)

    def build(self, num_channels):
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = self.x_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (self.kernel_size - 1) * dilation_size
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout_p=self.dropout_p,
                )
            ]
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(num_channels[-1], self.y_dim)

    def forward(self, x):
        # x: (N, C_in, L) where N is batch size, C_in is input channels, L is sequence length
        x = x.permute(1, 2, 0)  # Permute to (batch_size, channels, seq_len)
        out = self.network(x)
        out = out[:, :, -1]  # Take the last time step
        out = self.output_layer(out)
        return out
