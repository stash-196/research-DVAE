# base_model.py
from torch import nn
import torch
from collections import OrderedDict


class BaseModel(nn.Module):
    def __init__(self, x_dim, activation, dropout_p, device):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.activation = self._get_activation(activation)
        self.dropout_p = dropout_p
        self.device = device

    def forward(self, x):
        raise NotImplementedError("Each model must implement its own forward method.")

    def build(self):
        raise NotImplementedError("Each model must implement its own build method.")

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError("Unsupported activation type!")

    def _build_sequential_layers(self, layer_dims, input_dim):
        layers = OrderedDict()
        for idx, output_dim in enumerate(layer_dims):
            layers[f"linear{idx}"] = nn.Linear(input_dim, output_dim)
            layers[f"activation{idx}"] = self.activation
            layers[f"dropout{idx}"] = nn.Dropout(p=self.dropout_p)
            input_dim = output_dim
        if not layers:
            layers["Identity"] = nn.Identity()
        return nn.Sequential(layers), input_dim

    def prepare_mode_selector(self, mode_selector, seq_len, batch_size, x_dim, device):
        if mode_selector is not None:
            if not torch.is_tensor(mode_selector):
                mode_selector = torch.tensor(mode_selector, device=device)
            else:
                mode_selector = mode_selector.to(device)
        else:
            mode_selector = torch.zeros(seq_len, device=device)

        if mode_selector.dim() == 0:
            # Scalar mode_selector, expand to (seq_len, batch_size, x_dim)
            mode_selector = (
                mode_selector.view(1, 1, 1).expand(seq_len, batch_size, x_dim).float()
            )
        elif mode_selector.dim() == 1:
            # mode_selector of shape (seq_len,)
            mode_selector = (
                mode_selector.view(seq_len, 1, 1)
                .expand(seq_len, batch_size, x_dim)
                .float()
            )
        elif mode_selector.dim() == 2:
            # mode_selector of shape (seq_len, batch_size)
            mode_selector = (
                mode_selector.view(seq_len, batch_size, 1)
                .expand(seq_len, batch_size, x_dim)
                .float()
            )
        else:
            raise ValueError(f"Unsupported mode_selector shape: {mode_selector.shape}")

        return mode_selector

    def handle_nans(self, input_t, y_t, t):
        if input_t.isnan().any():
            if t == 0:
                input_t = torch.where(
                    input_t.isnan(), torch.zeros_like(input_t), input_t
                )
            else:
                input_t = torch.where(input_t.isnan(), y_t, input_t)
        return input_t
