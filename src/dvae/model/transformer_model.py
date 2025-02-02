# transformer_model.py
import torch
import torch.nn as nn
from base_model import BaseModel


class TransformerModel(BaseModel):
    def __init__(
        self,
        x_dim,
        y_dim,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        dropout,
        device,
    ):
        super().__init__(x_dim, y_dim, device)
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        self.build()

    def build(self):
        self.transformer = nn.Transformer(
            d_model=self.x_dim,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        self.input_proj = nn.Linear(self.x_dim, self.x_dim)
        self.output_proj = nn.Linear(self.x_dim, self.y_dim)

    def forward(self, src, tgt):
        # src: (S, N, E) where S is the source sequence length, N is the batch size, E is the feature number
        # tgt: (T, N, E) where T is the target sequence length

        src = self.input_proj(src)
        tgt = self.input_proj(tgt)

        output = self.transformer(src, tgt)
        output = self.output_proj(output)
        return output
