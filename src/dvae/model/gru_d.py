#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRU-D (Che et al., Scientific Reports 2018).

Multivariate GRU with two decays driven by the missingness mask and the
time since the last real observation:

* input decay pulls the carried-forward value toward the empirical mean
* hidden-state decay shrinks ``h`` over the same gap

The mask is concatenated into the GRU gates (the paper's ``V m`` terms).
NaNs in ``x`` are missing values. Do not pre-impute them.

Series layout matches the rest of this repo: ``(seq_len, batch, x_dim)``.
``mode_selector`` uses the same convention as the OTF RNNs (0 = teacher
forcing, 1 = closed loop). Closed-loop steps feed the model's own
prediction back in and mark that step observed, so the decay does not
treat a forecast as a missing measurement.

``noise_selector`` is accepted and ignored. GRU-D does not use the OTF
noise mixer.

This module is not wired into ``LearningAlgorithm``. Training and eval go
through ``bin/train_gru_d.py`` and ``bin/eval_gru_d.py``, which reuse the
existing dataset builder. To opt the live trainer in later, two additive
branches are enough (do not add them while that file is in use):

    elif self.model_name == "GRU_D":
        self.model = build_GRU_D(cfg=self.cfg, device=device)

and, in both the train and validation loss blocks,

    elif self.model_name in ("RNN", "MT_RNN", "GRU_D"):
        loss_kl = torch.zeros(1, device=self.device)

``forward`` already accepts ``mode_selector``, ``noise_selector``,
``logger``, and ``from_instance``.

PhysioNet 2012: point a GRU-D ini at ``dataset_name = PhysioNet2012`` and
``observation_process = raw_vitals`` (or ``raw_hr``). The hourly loader
leaves empty hours as NaN, which this cell reads as the mask. Keep
``delta_step = 1`` (one hour per row). Do not use the ``_indicate`` or
``_interpolate`` observation suffixes: the first doubles the width with a
mask channel GRU-D already builds internally, and the second fills the
gaps the decay is meant to see.
"""

from __future__ import annotations

import torch
from torch import nn

from dvae.model.base_model import BaseModel


def _int_list(cfg, section, option):
    if not cfg.has_option(section, option):
        return []
    raw = cfg.get(section, option).strip()
    if raw == "":
        return []
    return [int(piece) for piece in raw.split(",") if piece.strip()]


def masked_mse(pred, target):
    """Mean squared error on finite target entries.

    Missing targets (NaN) contribute nothing. If a batch has no finite
    targets the returned scalar is zero and still attached to ``pred``.
    """
    finite = torch.isfinite(target)
    if not torch.any(finite):
        return pred.sum() * 0.0
    safe_target = torch.where(finite, target, pred)
    err = (pred - safe_target).pow(2)
    return err[finite].mean()


class GRUD(BaseModel):
    """Single-layer GRU-D.

    ``num_rnn`` must be 1. The paper's cell is one layer; stacking is not
    implemented, so a larger value is an error rather than a silent ignore.
    """

    def __init__(
        self,
        x_dim,
        dim_rnn,
        dense_h_x,
        dropout_p=0.0,
        activation="relu",
        learn_x_mean=False,
        delta_step=1.0,
        device="cpu",
        cfg=None,
    ):
        super().__init__(x_dim, activation, dropout_p, device, cfg)
        self.dim_rnn = int(dim_rnn)
        self.dense_h_x = list(dense_h_x)
        self.learn_x_mean = bool(learn_x_mean)
        self.delta_step = float(delta_step)
        if self.delta_step <= 0:
            raise ValueError("delta_step must be positive.")
        self.type_rnn = "GRU_D"
        self.num_rnn = 1
        self.build()

    def build(self):
        d = self.x_dim
        h = self.dim_rnn
        # Diagonal-capable full maps, matching Che et al.: gamma = exp(-relu(W delta + b)).
        self.input_decay = nn.Linear(d, d)
        self.hidden_decay = nn.Linear(d, h)
        self.weight_xz = nn.Linear(d, h)
        self.weight_hz = nn.Linear(h, h, bias=False)
        self.weight_mz = nn.Linear(d, h, bias=False)
        self.weight_xr = nn.Linear(d, h)
        self.weight_hr = nn.Linear(h, h, bias=False)
        self.weight_mr = nn.Linear(d, h, bias=False)
        self.weight_xh = nn.Linear(d, h)
        self.weight_hh = nn.Linear(h, h, bias=False)
        self.weight_mh = nn.Linear(d, h, bias=False)
        self.mlp_h_x, last_dim = self._build_sequential_layers(self.dense_h_x, h)
        self.gen_out = nn.Linear(last_dim, self.y_dim)
        self.dropout = nn.Dropout(p=self.dropout_p)
        if self.learn_x_mean:
            self.x_mean = nn.Parameter(torch.zeros(d))
        else:
            self.register_buffer("x_mean", torch.zeros(d))
        self.reset_decay_parameters()
        self._clear_carry()

    def reset_decay_parameters(self):
        """Start with gamma = 1 (carry the last value; no decay) and let training move it."""
        nn.init.zeros_(self.input_decay.weight)
        nn.init.zeros_(self.input_decay.bias)
        nn.init.zeros_(self.hidden_decay.weight)
        nn.init.zeros_(self.hidden_decay.bias)

    def _clear_carry(self):
        self.y = None
        self.h = None
        self.h_t = None
        self.x_last = None
        self.delta = None

    def set_empirical_mean(self, mean):
        """Copy a per-coordinate observed mean into the decay target.

        ``mean`` is shape ``(x_dim,)``. When ``learn_x_mean`` is set this is
        only the initial value.
        """
        mean_t = torch.as_tensor(mean, dtype=self.x_mean.dtype, device=self.x_mean.device)
        if mean_t.shape != self.x_mean.shape:
            raise ValueError(
                f"empirical mean shape {tuple(mean_t.shape)} != {(self.x_dim,)}"
            )
        with torch.no_grad():
            self.x_mean.copy_(mean_t)

    def input_decay_gamma(self, delta):
        """``exp(-relu(W_x delta + b))``, shape ``(batch, x_dim)``."""
        return torch.exp(-torch.relu(self.input_decay(delta)))

    def hidden_decay_gamma(self, delta):
        """``exp(-relu(W_h delta + b))``, shape ``(batch, dim_rnn)``."""
        return torch.exp(-torch.relu(self.hidden_decay(delta)))

    def decayed_input(self, x_last, gamma_x):
        """Blend the carried value toward ``x_mean`` as gamma falls."""
        mean = self.x_mean.view(1, -1).to(dtype=x_last.dtype)
        return gamma_x * x_last + (1.0 - gamma_x) * mean

    def _gru_step(self, x_in, mask, h):
        z = torch.sigmoid(self.weight_xz(x_in) + self.weight_hz(h) + self.weight_mz(mask))
        r = torch.sigmoid(self.weight_xr(x_in) + self.weight_hr(h) + self.weight_mr(mask))
        h_tilde = torch.tanh(
            self.weight_xh(x_in) + self.weight_hh(r * h) + self.weight_mh(mask)
        )
        # Paper: h = (1 - z) * h_decayed + z * h_tilde. ``z`` opens toward the candidate.
        return (1.0 - z) * h + z * h_tilde

    def generation_x(self, h_t):
        return self.gen_out(self.mlp_h_x(h_t))

    def base_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x,
        initialize_states=True,
        mode_selector=None,
        noise_selector=None,
        inference_mode=False,
        logger=None,
        from_instance=None,
        mask=None,
    ):
        """
        Args:
            x: ``(seq_len, batch, x_dim)``. NaN marks a missing entry unless
                ``mask`` is passed.
            mask: optional ``(seq_len, batch, x_dim)`` boolean or 0/1 tensor.
                1 = observed. When omitted, the mask is ``isfinite(x)``.
            mode_selector: 0 teacher-forces the (decay-imputed) input, 1 feeds
                the current prediction back. Fractional values blend the two.
            noise_selector: ignored.

        Returns:
            ``y`` of shape ``(seq_len, batch, x_dim)``. ``y[t]`` is read from
            the decayed hidden state *before* ``x[t]`` is consumed, matching
            ``BaseRNN`` so the reconstruction loss is a one-step prediction.
        """
        del noise_selector, inference_mode, logger, from_instance
        if x.dim() != 3:
            raise ValueError(f"GRU-D expects (seq_len, batch, x_dim), got {tuple(x.shape)}")
        seq_len, batch_size, x_dim = x.shape
        if x_dim != self.x_dim:
            raise ValueError(f"x_dim {x_dim} does not match the model ({self.x_dim})")
        x = x.to(self.device)
        if mask is None:
            observed = torch.isfinite(x)
        else:
            observed = mask.to(self.device).bool()
            if observed.shape != x.shape:
                raise ValueError(
                    f"mask shape {tuple(observed.shape)} != x shape {tuple(x.shape)}"
                )
        x_filled = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        mode = self.prepare_mode_selector(
            mode_selector, seq_len, batch_size, self.x_dim, self.device
        )

        if initialize_states or self.h_t is None:
            h = torch.zeros(batch_size, self.dim_rnn, device=self.device, dtype=x.dtype)
            x_last = self.x_mean.view(1, -1).expand(batch_size, -1).to(dtype=x.dtype).clone()
            delta = torch.zeros(batch_size, self.x_dim, device=self.device, dtype=x.dtype)
        else:
            h = self.h_t
            x_last = self.x_last
            delta = self.delta

        y = torch.zeros(seq_len, batch_size, self.y_dim, device=self.device, dtype=x.dtype)
        h_seq = torch.zeros(seq_len, batch_size, self.dim_rnn, device=self.device, dtype=x.dtype)
        step = x.new_full((), self.delta_step)

        for t in range(seq_len):
            gamma_x = self.input_decay_gamma(delta)
            gamma_h = self.hidden_decay_gamma(delta)
            h_decayed = gamma_h * h
            y_t = self.generation_x(self.dropout(h_decayed))
            y[t] = y_t
            h_seq[t] = h_decayed

            m_t = observed[t].to(dtype=x.dtype)
            x_hat = m_t * x_filled[t] + (1.0 - m_t) * self.decayed_input(x_last, gamma_x)
            # Closed loop: feed this step's prediction and treat it as observed.
            auto = mode[t]
            x_in = (1.0 - auto) * x_hat + auto * y_t
            m_in = (1.0 - auto) * m_t + auto
            h = self._gru_step(x_in, m_in, h_decayed)

            fed = m_in > 0.5
            x_last = torch.where(fed, x_in, x_last)
            delta = torch.where(fed, step.expand_as(delta), delta + step)

        self.y = y
        self.h = h_seq
        self.h_t = h
        self.x_last = x_last
        self.delta = delta
        return y

    def get_info(self):
        info = [
            "----- GRU-D (Che et al. 2018) -----",
            f"x_dim={self.x_dim} dim_rnn={self.dim_rnn} delta_step={self.delta_step}",
            f"learn_x_mean={self.learn_x_mean}",
            "----- Input / hidden decay -----",
            str(self.input_decay),
            str(self.hidden_decay),
            "----- Readout -----",
        ]
        for layer in self.mlp_h_x:
            info.append(str(layer))
        info.append(str(self.gen_out))
        return info


def build_GRU_D(cfg, device="cpu"):
    """Build GRU-D from the same ini layout as the RNN configs."""
    x_dim = cfg.getint("Network", "x_dim")
    dim_rnn = cfg.getint("Network", "dim_rnn")
    num_rnn = cfg.getint("Network", "num_rnn", fallback=1)
    if num_rnn != 1:
        raise ValueError(
            "GRU-D implements the single-layer Che et al. cell; set num_rnn = 1."
        )
    activation = cfg.get("Network", "activation", fallback="relu")
    dropout_p = cfg.getfloat("Network", "dropout_p", fallback=0.0)
    dense_h_x = _int_list(cfg, "Network", "dense_h_x")
    learn_x_mean = cfg.getboolean("Training", "learn_x_mean", fallback=False)
    delta_step = cfg.getfloat("Training", "delta_step", fallback=1.0)
    model = GRUD(
        x_dim=x_dim,
        dim_rnn=dim_rnn,
        dense_h_x=dense_h_x,
        dropout_p=dropout_p,
        activation=activation,
        learn_x_mean=learn_x_mean,
        delta_step=delta_step,
        device=device,
        cfg=cfg,
    ).to(device)
    return model
