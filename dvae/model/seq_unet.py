import torch
import torch.nn as nn
from base_model import BaseModel


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dropout_p):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class SeqUNetModel(BaseModel):
    def __init__(self, x_dim, y_dim, num_channels, kernel_size, dropout_p, device):
        super().__init__(x_dim, y_dim, device)
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p

        self.build(num_channels)

    def build(self, num_channels):
        # Encoder
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        num_levels = len(num_channels)
        for i in range(num_levels):
            in_channels = self.x_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            self.encoders.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    self.kernel_size,
                    padding=1,
                    dropout_p=self.dropout_p,
                )
            )

        # Decoder
        for i in reversed(range(num_levels - 1)):
            in_channels = num_channels[i + 1]
            out_channels = num_channels[i]
            self.decoders.append(
                ConvBlock(
                    in_channels * 2,
                    out_channels,
                    self.kernel_size,
                    padding=1,
                    dropout_p=self.dropout_p,
                )
            )

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.final_conv = nn.Conv1d(num_channels[0], self.y_dim, kernel_size=1)

    def forward(self, x):
        # x: (N, C_in, L)
        x = x.permute(0, 2, 1)  # Convert to (N, L, C_in)
        enc_outs = []

        # Encoder
        for encoder in self.encoders:
            x = encoder(x)
            enc_outs.append(x)
            x = self.maxpool(x)

        # Bottleneck
        x = enc_outs[-1]

        # Decoder
        for idx, decoder in enumerate(self.decoders):
            x = self.upsample(x)
            x = torch.cat([x, enc_outs[-(idx + 2)]], dim=1)
            x = decoder(x)

        x = self.final_conv(x)
        x = x.permute(0, 2, 1)  # Convert back to (N, L, C_out)
        return x
