# base_vrnn.py
from torch import nn
import torch
from collections import OrderedDict
from dvae.model.base_model import BaseModel


class BaseVRNN(BaseModel):
    def __init__(
        self,
        x_dim,
        z_dim,
        activation,
        dense_x,
        dense_z,
        dense_hx_z,
        dense_hz_x,
        dense_h_z,
        dim_rnn,
        num_rnn,
        type_rnn,
        dropout_p,
        device,
    ):
        super().__init__(x_dim, activation, dropout_p, device)
        self.z_dim = z_dim
        # Feature extractors
        self.dense_x = dense_x
        self.dense_z = dense_z
        # Dense layers for generation
        self.dense_hx_z = dense_hx_z
        self.dense_hz_x = dense_hz_x
        self.dense_h_z = dense_h_z
        # RNN
        self.dim_rnn = dim_rnn
        self.num_rnn = num_rnn
        self.type_rnn = type_rnn
        self.build()

    def build(self):
        # Feature extractors
        self.feature_extractor_x, self.dim_feature_x = self._build_sequential_layers(
            self.dense_x, self.x_dim
        )
        self.feature_extractor_z, self.dim_feature_z = self._build_sequential_layers(
            self.dense_z, self.z_dim
        )

        # Inference network
        self.mlp_hx_z, dim_hx_z = self._build_sequential_layers(
            self.dense_hx_z, self.dim_rnn + self.dim_feature_x
        )
        self.inf_mean = nn.Linear(dim_hx_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_hx_z, self.z_dim)

        # Prior network
        self.mlp_h_z, dim_h_z = self._build_sequential_layers(
            self.dense_h_z, self.dim_rnn
        )
        self.prior_mean = nn.Linear(dim_h_z, self.z_dim)
        self.prior_logvar = nn.Linear(dim_h_z, self.z_dim)

        # Generative network
        self.mlp_hz_x, dim_hz_x = self._build_sequential_layers(
            self.dense_hz_x, self.dim_rnn + self.dim_feature_z
        )
        self.gen_out = nn.Linear(dim_hz_x, self.y_dim)

        # Recurrence
        self.rnn = self._build_recurrence()

    def _build_recurrence(self):
        input_dim = self.dim_feature_x + self.dim_feature_z
        if self.type_rnn == "LSTM":
            return nn.LSTM(input_dim, self.dim_rnn, self.num_rnn)
        elif self.type_rnn == "RNN":
            return nn.RNN(input_dim, self.dim_rnn, self.num_rnn)
        else:
            raise ValueError("Unsupported RNN type!")

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def generation_x(self, feature_zt, h_t):
        dec_input = torch.cat((feature_zt, h_t), -1)
        assert torch.isnan(dec_input).sum() == 0, "NaNs in dec_input"
        dec_output = self.mlp_hz_x(dec_input)
        assert torch.isnan(dec_output).sum() == 0, "NaNs in dec_output"
        y_t = self.gen_out(dec_output)
        assert torch.isnan(y_t).sum() == 0, "NaNs in gen_out"
        return y_t

    # Generate the Prior Distribution of z
    def generation_z(self, h_t):
        prior_output = self.mlp_h_z(h_t)
        mean_prior = self.prior_mean(prior_output)
        logvar_prior = self.prior_logvar(prior_output)
        return mean_prior, logvar_prior

    # Generate the Posterior Distribution of z
    def inference(self, feature_xt, h_t):
        enc_input = torch.cat((feature_xt, h_t), -1)
        enc_output = self.mlp_hx_z(enc_input)
        mean_zt = self.inf_mean(enc_output)
        logvar_zt = self.inf_logvar(enc_output)
        return mean_zt, logvar_zt

    def recurrence(self, feature_xt, feature_zt, h_t, c_t=None):
        # rnn_input Shape: (1, batch_size, input_size)
        rnn_input = torch.cat((feature_xt, feature_zt), -1)
        if self.type_rnn == "LSTM":
            _, (h_tp1, c_tp1) = self.rnn(rnn_input, (h_t, c_t))
            return h_tp1, c_tp1
        elif self.type_rnn == "RNN":
            _, h_tp1 = self.rnn(rnn_input, h_t)
            return h_tp1, None

    def forward(
        self,
        x,
        initialize_states=True,
        mode_selector=None,
        inference_mode=False,
        logger=None,
        from_instance=None,
    ):
        seq_len, batch_size, _ = x.shape

        if initialize_states:
            # Initialize variables
            feature_x = torch.zeros(
                (seq_len, batch_size, self.dense_x[-1] if self.dense_x else self.x_dim),
                device=self.device,
            )

            y = torch.zeros((seq_len, batch_size, self.y_dim), device=self.device)
            h = torch.zeros((seq_len, batch_size, self.dim_rnn), device=self.device)
            h_t = torch.zeros(
                self.num_rnn, batch_size, self.dim_rnn, device=self.device
            )
            if self.type_rnn == "LSTM":
                c_t = torch.zeros(
                    self.num_rnn, batch_size, self.dim_rnn, device=self.device
                )

            z_mean = torch.zeros((seq_len, batch_size, self.z_dim), device=self.device)
            z_logvar = torch.zeros(
                (seq_len, batch_size, self.z_dim), device=self.device
            )
            z = torch.zeros((seq_len, batch_size, self.z_dim), device=self.device)
            z_t = torch.randn((batch_size, self.z_dim), device=self.device)

        else:
            y = self.y
            h = self.h
            h_t = self.h_t
            if self.type_rnn == "LSTM":
                c_t = self.c_t
            z = self.z
            z_mean = self.z_mean
            z_logvar = self.z_logvar
            z_t = self.z_t

        # Prepare mode selector
        mode_selector = self.prepare_mode_selector(
            mode_selector, seq_len, batch_size, self.x_dim, self.device
        )
        for t in range(seq_len):
            input_t = x[t, :, :]  # Shape: (batch_size, x_dim)
            mode_selector_t = mode_selector[t, :, :]  # Shape: (batch_size, x_dim)

            # Impute NaNs with previous output
            imputed_input_t = self.impute_inputs_nans_with_output(
                input_t, y_t if t > 0 else None, t
            )

            # Mix inputs based on mode_selector and extract features
            mixed_input_t = self.mix_inputs_with_outputs(
                imputed_input_t, y_t if t > 0 else None, mode_selector_t, t
            )

            feature_xt = self.feature_extractor_x(mixed_input_t)

            # Get the last hidden state
            h_t_last = h_t[-1]  # Shape: (batch_size, dim_rnn)

            # Inference and reparameterization
            mean_zt, logvar_zt = self.inference(feature_xt, h_t_last)
            if inference_mode:
                z_t = mean_zt
            else:
                z_t = self.reparameterization(mean_zt, logvar_zt)
            feature_zt = self.feature_extractor_z(z_t)

            # Generation
            y_t = self.generation_x(feature_zt, h_t_last)
            assert (
                torch.isnan(y_t).sum() == 0
            ), f"NaNs in y_t at t={t}, {torch.isnan(y_t).sum()} NaNs"
            y[t, :, :] = y_t
            feature_x[t, :, :] = feature_xt
            h[t, :, :] = h_t_last

            z_mean[t, :, :] = mean_zt
            z_logvar[t, :, :] = logvar_zt
            z[t, :, :] = z_t

            # Recurrence
            if self.type_rnn == "LSTM":
                h_t, c_t = self.recurrence(
                    feature_xt.unsqueeze(0), feature_zt.unsqueeze(0), h_t, c_t
                )
            elif self.type_rnn == "RNN":
                h_t, _ = self.recurrence(
                    feature_xt.unsqueeze(0), feature_zt.unsqueeze(0), h_t
                )

        z_mean_p, z_logvar_p = self.generation_z(h)

        # Save states
        self.feature_x = feature_x
        self.y = y
        self.h = h
        self.h_t = h_t
        if self.type_rnn == "LSTM":
            self.c_t = c_t

        self.z = z
        self.z_t = z_t
        self.z_mean = z_mean
        self.z_logvar = z_logvar
        self.z_mean_p = z_mean_p
        self.z_logvar_p = z_logvar_p

        return y

    def get_info(self):

        info = []
        info.append("----- Feature extractor -----")
        for layer in self.feature_extractor_x:
            info.append(str(layer))
        for layer in self.feature_extractor_z:
            info.append(str(layer))
        info.append("----- Inference -----")
        for layer in self.mlp_hx_z:
            info.append(str(layer))
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))
        info.append("----- Generation x -----")
        for layer in self.mlp_hz_x:
            info.append(str(layer))
        info.append(str(self.gen_out))
        info.append("----- Recurrence -----")
        info.append(str(self.rnn))
        info.append("----- Generation z -----")
        for layer in self.mlp_h_z:
            info.append(str(layer))
        info.append(str(self.prior_mean))
        info.append(str(self.prior_logvar))

        return info
