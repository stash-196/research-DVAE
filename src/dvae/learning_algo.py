#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt
"""

import os
import warnings
import sys
import shutil
import socket
import datetime
import time
import numpy as np
from typing import Optional
import torch
import matplotlib.pyplot as plt
from dvae.utils import (
    find_project_root,
    myconf,
    get_logger,
    loss_ISD,
    loss_KLD,
    loss_MPJPE,
    loss_MSE,
    create_autonomous_mode_selector,
    profile_execution,
)
from dvae.visualizers import (
    visualize_model_parameters,
    visualize_combined_parameters,
    visualize_teacherforcing_2_autonomous,
    visualize_total_loss,
    visualize_val_criteria,
    visualize_recon_loss,
    visualize_kld_loss,
    visualize_combined_metrics,
    visualize_sigma_history,
    visualize_alpha_history,
    save_drift_timestep_figures,
)

from dvae.dataset.dataset_builder import build_dataloader, DatasetConfig
from dvae.dataset.xhro_proper_dataset import (
    coarse_dataframe_kwargs,
    optional_config_bool,
    optional_config_float,
    optional_config_str,
    training_initial_sequence_len,
    windowing_from_config,
)
from dvae.eval.utils.local_drift_analysis import (
    DRIFT_METRIC_NAMES,
    DRIFT_PLOT_NAMES,
    RAW_DRIFT_METRICS,
    VARNORM_DRIFT_METRICS,
    enforce_drift_span,
    rosenstein_on_free_run,
    score_fixed_validation_windows,
    x_true_variance,
)
from dvae.eval.utils.delay_params import _get_delay_params
from dvae.eval.utils.spectrum_validation import (
    capture_contiguous_val_windows,
    capture_fixed_spectrum_windows,
)
from dvae.eval.window_length import cap_eval_seq_len
from dvae.epoch_log import (
    format_bump_line,
    format_early_stop_line,
    format_epoch_budget_line,
    format_new_best_line,
    format_selection_line,
    format_timing_line,
    make_bump_row,
)
from dvae.wall_clock import sync_cuda
from dvae.regularizers import (
    parse_regularizer_settings,
    regularizer_losses,
    warmup_progress,
)
from dvae.warmup_curriculum import (
    WarmupCurriculum,
    format_patience,
    warmup_action,
    warmup_settings_from_config,
)
from dvae.run_options import (
    DEFAULT_TF_MAX_GAP_FRAC,
    DEFAULT_TF_MIN_OBSERVED,
    canonical_criterion_name,
    criterion_series_key,
    normalised_criterion_sum,
    selection_criterion_value,
    parse_checkpoint_every_n_epochs,
    parse_optional_int,
    parse_unit_interval,
    parse_val_drift_horizon,
    parse_val_rosenstein,
    legacy_defaults_from_config,
    resolve_ftle_fit,
    resolve_val_criteria,
    spectrum_monitoring_enabled,
)
from dvae.model import (
    build_VRNN,
    build_RNN,
    build_MT_RNN,
    build_MT_VRNN,
)
from dvae.dataset.gap_grid import snapshot_window_stats
from dvae.training_history import (
    build_training_history,
    drift_series_source,
    restore_per_window,
    restore_matrix,
    save_training_history,
    series_or_new,
    warmup_change_epochs,
)
from dvae.timescale_history import (
    init_A_sigmas_history,
    log_plrnn_timescales,
    save_plrnn_timescale_figures,
    update_A_sigmas_history,
)
import subprocess


def selection_recon_value(algo, epoch, val_loss) -> float:
    """Recon used for selection and early stopping.

    The fixed-window recon (teacher-forced MSE on the fixed validation
    windows) when the contiguous validation stretch is on, else the
    loader ``val_loss`` as before.
    """
    fixed = getattr(algo, "_val_fixed_recon", None)
    if getattr(algo, "_contiguous_val", False) and fixed is not None:
        return float(fixed[epoch])
    return float(val_loss[epoch])


class LearningAlgorithm:
    def __init__(self, params):
        # Load parameters
        self.params = params
        self.config_file = self.params["cfg"]
        if not os.path.isfile(self.config_file):
            raise ValueError("Invalid config file path")
        self.cfg = myconf()
        self.cfg.read(self.config_file)
        self.experiment_name = self.cfg.get("User", "experiment_name")
        self.job_id = self.params["job_id"]
        print(f"[Learning Algo] Job ID: {self.job_id}")
        self.model_name = self.cfg.get("Network", "name")
        self.dataset_name = self.cfg.get("DataFrame", "dataset_name")
        self.dataset_label = self.cfg.get("DataFrame", "dataset_label", fallback="None")
        self.mask_label = self.cfg.get("DataFrame", "mask_label", fallback="None")
        self.sequence_len = self.cfg.getint("DataFrame", "sequence_len")
        self.shuffle = self.cfg.getboolean("DataFrame", "shuffle")
        self.num_workers = self.cfg.getint("DataFrame", "num_workers")
        self.batch_size = self.cfg.getint("DataFrame", "batch_size")

        # Get device-specific paths
        self.device_config = self.params["device_config"]
        self.saved_root = self.device_config["saved_root"]
        self.data_dir = self.device_config["data_dir"]
        self.device_name = self.device_config["device_name"]

        # Get training parameters
        self.autonomous_sampling_method = self.cfg.get("Training", "sampling_method")
        self.sampling_ratio = self.cfg.getfloat("Training", "sampling_ratio")
        self.kl_warm_limit = self.cfg.getfloat(
            "Training", "kl_warm_limit", fallback=1.0
        )
        self.auto_warm_start = self.cfg.getfloat(
            "Training", "auto_warm_start", fallback=0.0
        )
        self.auto_warm_limit = self.sampling_ratio
        self.mask_out_autonomous_from_loss = self.cfg.getboolean(
            "Training", "mask_autonomous_filled", fallback=False
        )
        self.loss_mask_mode = self.cfg.get(
            "Training", "loss_mask_mode", fallback="none"
        )
        self.noise_warm_limit = self.cfg.getfloat(
            "Training", "noise_max_ratio", fallback=1.0
        )
        self.noise_warm_start = self.cfg.getfloat(
            "Training", "noise_init_ratio", fallback=1.0
        )
        self.noise_ratio = self.cfg.getfloat(
            "Training", "noise_mix_ratio", fallback=0.0
        )
        self.noise_warm_reset_on_window = self.cfg.getboolean(
            "Training", "noise_warm_reset_on_window", fallback=True
        )  # True: reset noise on seq_len bump
        self.tie_noise_to_auto = self.cfg.getboolean(
            "Training", "tie_noise_to_auto", fallback=False
        )  # True: force noise_warm = auto_warm (for tied ablations)
        self.enable_autonomous_warmup = self.autonomous_sampling_method == "ss"
        self.enable_noise_warmup = self.enable_autonomous_warmup and self.cfg.get(
            "Training", "noise_sampling_method", fallback="none"
        ) in ["ss", "sm"]
        self.enable_kl_warmup = self.model_name in ["VRNN", "MT_VRNN"]

        # Get host name and date
        self.hostname = socket.gethostname()
        self.datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H.%M.%S")
        self.date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        self.time_str = datetime.datetime.now().strftime("%H.%M.%S")

        # Load model parameters
        self.use_cuda = self.cfg.getboolean("Training", "use_cuda")
        if torch.cuda.is_available() and self.use_cuda:
            self.device = "cuda"
        elif torch.backends.mps.is_available() and self.use_cuda:
            self.device = "mps"
        else:
            self.device = "cpu"

        try:
            self.optimize_alphas = self.cfg.getboolean("Training", "optimize_alphas")
        except ValueError:
            self.optimize_alphas = None

        if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
            self.alphas = [
                float(i)
                for i in self.cfg.get("Network", "alphas").split(",")
                if i != ""
            ]

    def build_model(self, device=None):
        if device is None:
            device = self.device

        if self.model_name == "VRNN":
            self.model = build_VRNN(cfg=self.cfg, device=device)
        elif self.model_name == "RNN":
            self.model = build_RNN(cfg=self.cfg, device=device)
        elif self.model_name == "MT_RNN":
            self.model = build_MT_RNN(cfg=self.cfg, device=device)
        elif self.model_name == "MT_VRNN":
            self.model = build_MT_VRNN(cfg=self.cfg, device=device)

    def init_optimizer(self):
        optimization = self.cfg.get("Training", "optimization")
        self.lr = self.cfg.getfloat("Training", "lr")
        if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
            # defaults to the same as lr if not present
            self.alpha_lr = self.cfg.getfloat("Training", "alpha_lr", fallback=self.lr)
            # base_parameters drops every name that contains "sigmas",
            # so this group is the only one that steps the timescale.
            params = [
                {"params": self.model.base_parameters(), "lr": self.lr},
                {"params": [self.model.sigmas], "lr": self.alpha_lr},
            ]
        else:
            params = self.model.parameters()

        if optimization == "adam":
            optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=1e-5)
        elif optimization == "adamW":
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-5)
        else:
            # fallback to Adam if optimization method is not recognized
            optimizer = torch.optim.Adam(params, lr=self.lr)

        return optimizer

    def get_basic_info(self):
        basic_info = []
        basic_info.append("HOSTNAME: " + self.hostname)
        basic_info.append("Time: " + self.datetime_str)
        basic_info.append("Device for training: " + self.device)
        if self.device == "cuda":
            basic_info.append("Cuda verion: {}".format(torch.version.cuda))
        basic_info.append("Model name: {}".format(self.model_name))
        basic_info.append("Total params: {:,}".format(self.model.parameter_count()))

        return basic_info

    def _prepare_masked_loss(
        self,
        batch_data,
        recon_batch_data,
        loss_mask,
        model_mode_selector,
        dataset_config,
    ):
        """
        Helper for masking baseline (Che et al. GRU-D, Lipton et al. 2016):
        When observation_process == "only_x_indicate", enforce:
        - 100% teacher forcing on the mask channel (dimension 1)
        - Loss computed only on the signal channel (dimension 0)

        Args:
            batch_data: (seq_len, batch_size, x_dim) ground truth
            recon_batch_data: (seq_len, batch_size, x_dim) reconstruction
            loss_mask: (seq_len, batch_size, x_dim) masking indicator (1 = compute loss, 0 = ignore)
            model_mode_selector: (seq_len, batch_size, x_dim) mode selector (0 = TF, 1 = autonomous)
            dataset_config: dataset configuration object

        Returns:
            loss_recon: scalar reconstruction loss (only on observed signal values)
        """
        is_masking_baseline = self._is_indicate_observation(
            getattr(dataset_config, "observation_process", None)
        )

        if is_masking_baseline:
            # Indicate layout is interleaved [x0, m0, x1, m1, ...]:
            # force TF on mask channels (odd dims); loss only on signal (even dims).
            model_mode_selector = model_mode_selector.clone()
            model_mode_selector[:, :, 1::2] = 0.0

            target_signal = batch_data[:, :, 0::2]
            pred_signal = recon_batch_data[:, :, 0::2]
            loss_mask_signal = loss_mask[:, :, 0::2]
        else:
            target_signal = batch_data
            pred_signal = recon_batch_data
            loss_mask_signal = loss_mask

        # Apply mask and convert NaNs to zero
        target_masked = target_signal * loss_mask_signal
        target_masked = torch.nan_to_num(target_masked, nan=0.0)
        recon_masked = pred_signal * loss_mask_signal

        return loss_MSE(target_masked, recon_masked)

    def _is_indicate_observation(self, observation_process: Optional[str]) -> bool:
        """Return True if observation_process is an 'indicate' variant.

        Accepts exact 'only_x_indicate' or suffix forms like 'raw_ch1_indicate'.
        """
        if observation_process is None:
            return False
        if observation_process == "only_x_indicate":
            return True
        if isinstance(observation_process, str) and observation_process.endswith(
            "_indicate"
        ):
            # allow forms like raw_ch1_indicate where base may be in select_columns
            return True
        return False

    def _score_fixed_validation(self) -> dict:
        """Spectrum error and Appendix B drift on the fixed windows.

        Does not resize the live loader. The flip and the horizon are the
        values stored at training start.
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                return score_fixed_validation_windows(
                    self.model,
                    self._spectrum_windows,
                    tf_steps=self._spectrum_tf_steps,
                    horizon=self._drift_horizon,
                    variance=self._x_true_var,
                    observation_process=self._spectrum_observation,
                    device=self.device,
                    ftle_k_start=self._ftle_fit[0],
                    ftle_k_end=self._ftle_fit[1],
                    rosenstein=self._score_rosenstein,
                    embed_dim=self._embed_dim,
                    embed_tau=self._embed_tau,
                    strict_spectrum=bool(getattr(self, "_contiguous_val", False)),
                )
        finally:
            if was_training:
                self.model.train()

    def _selection_recon(self, epoch, val_loss) -> float:
        """See ``selection_recon_value``."""
        return selection_recon_value(self, epoch, val_loss)

    def _raw_criterion(self, name, epoch, val_loss, val_spectrum, drift_arrays):
        """Selection value for one criterion. Lower is better.

        ``recon`` is ``val_loss``. ``spectrum`` is the free-run spectral
        error. A cross term is ``|cross|``. ``ftle`` and ``rosenstein_lle``
        are ``|estimate - GT LLE|``. The stored series stay signed.
        """
        if name == "recon":
            raw = selection_recon_value(self, epoch, val_loss)
        elif name == "spectrum":
            raw = float(val_spectrum[epoch])
        else:
            raw = float(
                drift_arrays["val_" + criterion_series_key(name)][epoch]
            )
        return selection_criterion_value(name, raw, gt_lle=self._lle_gt)

    def _epoch_selection_value(
        self, epoch, val_loss, val_spectrum, val_criteria_score, drift_arrays
    ):
        """Score used for best-model selection and early stopping.

        ``recon`` alone returns ``val_loss[epoch]``. Any other single
        criterion returns its selection value: the stored series, except
        ``|cross|`` for a cross term and ``|estimate - GT LLE|`` for
        ``ftle`` and ``rosenstein_lle``. Several criteria return the sum
        of each of those values divided by its value at the first
        validation epoch. Spectrum and the Appendix B terms are filled
        whenever the fixed windows exist, including when they are not
        selected. Returns ``(score, raw components, normalised components)``.
        A single criterion has an empty normalised dict.
        """
        self._epoch_metric_times = {
            "time_val_forward": 0.0,
            "time_spectrum": 0.0,
            "time_drift": 0.0,
            "time_rosenstein": 0.0,
        }
        if self._spectrum_windows is not None:
            scored = self._score_fixed_validation()
            val_spectrum[epoch] = float(scored["spectrum"])
            if getattr(self, "_val_fixed_recon", None) is not None:
                self._val_fixed_recon[epoch] = float(scored["recon_fixed"])
                self.logger.info(
                    "Val fixed-window recon (teacher-forced, {} window(s)): "
                    "{:.6f}".format(len(self._spectrum_windows), self._val_fixed_recon[epoch])
                )
            skipped = int(scored.get("spectrum_skipped", 0))
            if self._val_spectrum_skipped is not None:
                self._val_spectrum_skipped[epoch] = skipped
            if skipped and getattr(self, "_contiguous_val", False):
                self.logger.warning(
                    "Val spectrum: {} (window, channel) free-run segment(s) "
                    "have too few finite samples. Spectrum is NaN this epoch "
                    "(no subset average).".format(skipped)
                )
            elif skipped:
                self.logger.warning(
                    "Val spectrum skipped {} (window, channel) segment(s) "
                    "with too few finite samples.".format(skipped)
                )
            for name in DRIFT_METRIC_NAMES:
                drift_arrays["val_" + name][epoch] = float(scored[name])
            if self._lle_per_window is not None:
                self._lle_per_window[epoch] = np.asarray(
                    scored["lle_per_window"], dtype=float
                )
                self._ftle_per_window[epoch] = np.asarray(
                    scored["ftle_per_window"], dtype=float
                )
            # Full-window traces for the checkpoint figure. The scalar
            # series above stay the post-flip means. These arrays are
            # replaced every epoch and are not stored in the checkpoint.
            self._latest_drift_steps = {
                "e2": np.asarray(scored["e2_per_step"], dtype=np.float64),
                "ed": np.asarray(scored["ed_per_step"], dtype=np.float64),
                "d2": np.asarray(scored["d2_per_step"], dtype=np.float64),
                "cos": np.asarray(scored["cos_per_step"], dtype=np.float64),
            }
            self._latest_drift_steps_varnorm = {
                "e2": np.asarray(scored["e2_per_step_varnorm"], dtype=np.float64),
                "ed": np.asarray(scored["ed_per_step_varnorm"], dtype=np.float64),
                "d2": np.asarray(scored["d2_per_step_varnorm"], dtype=np.float64),
                "cos": np.asarray(scored["cos_per_step_varnorm"], dtype=np.float64),
            }
            self.logger.info(
                "Val spectrum error (free-run half): {:.6f}".format(
                    val_spectrum[epoch]
                )
            )
            self.logger.info(
                "Val drift horizon {}: base_error {:.6f} auto_drift {:.6f} "
                "error_drift_cross {:.6f} error_drift_total {:.6f} "
                "recon_k {:.6f}".format(
                    self._drift_horizon,
                    scored["base_error"],
                    scored["auto_drift"],
                    scored["error_drift_cross"],
                    scored["error_drift_total"],
                    scored["recon_k"],
                )
            )
            self.logger.info(
                "Val drift / var: base_error {:.6f} auto_drift {:.6f} "
                "error_drift_cross {:.6f} error_drift_total {:.6f}".format(
                    scored["base_error_varnorm"],
                    scored["auto_drift_varnorm"],
                    scored["error_drift_cross_varnorm"],
                    scored["error_drift_total_varnorm"],
                )
            )
            self.logger.info(
                "Val auto_drift at k=1 {:.6f}, k=5 {:.6f}, k={} {:.6f}".format(
                    scored["auto_drift_k1"],
                    scored["auto_drift_k5"],
                    self._drift_horizon,
                    scored["auto_drift_k"],
                )
            )
            self.logger.info(
                "Val finite-time exponent (k={}..{}): {:.6f}".format(
                    self._ftle_fit[0], self._ftle_fit[1], scored["ftle"]
                )
            )
            if self._score_rosenstein:
                gt = self._lle_gt
                gt_text = "nan" if gt is None or not np.isfinite(gt) else "{:.6f}".format(gt)
                self.logger.info(
                    "Val Rosenstein LLE mean {:.6f} std {:.6f} (GT {})".format(
                        scored["lle_mean"], scored["lle_std"], gt_text
                    )
                )
            for key in (
                "time_val_forward",
                "time_spectrum",
                "time_drift",
                "time_rosenstein",
            ):
                self._epoch_metric_times[key] = float(scored.get(key, 0.0))
        if self._metric_reference is None:
            reference = {"recon": selection_recon_value(self, epoch, val_loss)}
            if self._spectrum_windows is not None:
                reference["spectrum"] = float(val_spectrum[epoch])
                for name in RAW_DRIFT_METRICS:
                    reference[name] = float(drift_arrays["val_" + name][epoch])
            self._metric_reference = reference
            self.logger.info(
                "Val metric reference (first validation epoch {}): {}".format(
                    epoch, self._metric_reference
                )
            )
        raw = {}
        for name in self._val_criteria:
            raw[name] = self._raw_criterion(
                name, epoch, val_loss, val_spectrum, drift_arrays
            )
        # Every spectrum segment skipped: this epoch is not eligible for
        # best-model selection and does not reset patience.
        spectrum_missing = "spectrum" in raw and not np.isfinite(raw["spectrum"])
        self._selection_skipped = bool(spectrum_missing)
        if spectrum_missing:
            self.logger.warning(
                "Epoch {}: selection skipped: spectrum NaN".format(epoch)
            )
        else:
            self._any_selection_eligible = True
        if len(self._val_criteria) == 1:
            return raw[self._val_criteria[0]], raw, {}
        if self._val_reference is None and not spectrum_missing:
            self._val_reference = dict(raw)
            self.logger.info(
                "Val criteria reference (first validation epoch {}): {}".format(
                    epoch, self._val_reference
                )
            )
        if spectrum_missing:
            # No score this epoch. The reference waits for the first epoch
            # with a finite spectrum.
            val_criteria_score[epoch] = float("nan")
            norm = {name: float("nan") for name in self._val_criteria}
            return float("nan"), raw, norm
        score = normalised_criterion_sum(raw, self._val_reference)
        val_criteria_score[epoch] = score
        norm = {
            name: raw[name] / float(self._val_reference[name])
            for name in self._val_criteria
        }
        return score, raw, norm

    def train(self, profiler=None):
        # Build model
        self.build_model()

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        project_root = find_project_root(__file__)

        # Create directory for results
        if not self.params["reload"]:
            tag = self.cfg.get("Network", "tag")
            type_rnn = self.cfg.get("Network", "type_rnn")
            # Define all the parts of the filename in a list
            filename_parts = [
                self.dataset_name,
                self.dataset_label,
                self.mask_label,
                tag,
                type_rnn,
                self.autonomous_sampling_method,
                self.sampling_ratio,
            ]

            # Convert all parts to strings and join them with an underscore
            # This is dynamic, readable, and efficient.
            filename = "_".join(
                str(part) for part in filename_parts if part is not None and part != ""
            )

            if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
                compressed_alphas = ",".join(
                    "{:.3f}".format(alpha) for alpha in self.alphas[:10]
                )
                if len(self.alphas) > 10:
                    compressed_alphas += ",..."
                filename += "_α{}".format(compressed_alphas)

            filename += "_{}_{}".format(self.datetime_str, self.job_id)

            save_dir = os.path.join(
                self.saved_root,
                self.date_str,
                f"{self.device_name}",
                f"{self.experiment_name}",
                filename,
            )
            print(f"[Learning Algo] Saving to: {save_dir}")
            try:
                os.makedirs(save_dir)
            except FileExistsError:
                print(f"[Learning Algo] Directory already exists: {save_dir}")
        else:
            tag = self.cfg.get("Network", "tag")
            save_dir = self.params["model_dir"]

        # Save the experiment configuration, unless src and dst are the same file.
        save_cfg = os.path.join(save_dir, "config.ini")
        src_cfg = os.path.abspath(self.config_file)
        dst_cfg = os.path.abspath(save_cfg)
        if not (os.path.exists(dst_cfg) and os.path.samefile(src_cfg, dst_cfg)):
            shutil.copy(src_cfg, dst_cfg)

        # Create logger
        log_file = os.path.join(save_dir, "log.txt")
        logger_type = self.cfg.getint("User", "logger_type")
        logger = get_logger(log_file, logger_type)
        self.logger = logger

        # Print basic information
        for log in self.get_basic_info():
            logger.info(log)
        logger.info("In this experiment, result will be saved in: " + save_dir)

        # Print model information (optional)
        if self.cfg.getboolean("User", "print_model"):
            for log in self.model.get_info():
                logger.info(log)

        # Init optimizer
        optimizer = self.init_optimizer()

        # Create initial data loader with the starting sequence length
        initial_sequence_len = training_initial_sequence_len(self.cfg)

        # Define DatasetConfig with all parameters
        dataset_config = DatasetConfig(
            data_dir=self.data_dir,
            x_dim=self.cfg.getint(
                "Network", "x_dim"
            ),  # Assume x_dim is set elsewhere in build_model or config
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            sample_rate=self.cfg.getint(
                "DataFrame", "sample_rate"
            ),  # Assume these are set from cfg
            skip_rate=self.cfg.getint("DataFrame", "skip_rate"),
            val_indices=self.cfg.getfloat("DataFrame", "val_indices"),
            observation_process=self.cfg.get("DataFrame", "observation_process"),
            overlap=self.cfg.getboolean("DataFrame", "overlap"),
            with_nan=self.cfg.getboolean("DataFrame", "with_nan", fallback=False),
            seq_len=initial_sequence_len,  # Use initial_sequence_len here
            device=self.device,
            dataset_label=self.dataset_label,
            mask_label=self.mask_label,
            data_root=optional_config_str(self.cfg, "DataFrame", "data_root"),
            corpus=optional_config_str(self.cfg, "DataFrame", "corpus"),
            crop_start_sec=optional_config_float(self.cfg, "DataFrame", "crop_start_sec"),
            crop_end_sec=optional_config_float(self.cfg, "DataFrame", "crop_end_sec"),
            segments=optional_config_str(self.cfg, "DataFrame", "segments"),
            max_fill_gap_sec=optional_config_float(
                self.cfg, "DataFrame", "max_fill_gap_sec"
            ),
            filled_nan_log_sec=optional_config_float(
                self.cfg, "DataFrame", "filled_nan_log_sec"
            ),
            **windowing_from_config(self.cfg),
            block_mask_file=optional_config_str(
                self.cfg, "DataFrame", "block_mask_file"
            ),
            block_mask_mode=optional_config_str(
                self.cfg, "DataFrame", "block_mask_mode"
            ),
            contiguous_val_split=optional_config_bool(
                self.cfg, "DataFrame", "contiguous_val_split", default=False
            ),
            **coarse_dataframe_kwargs(self.cfg),
        )

        # Build data loaders (no need to pass sequence_len and device separately)
        train_dataloader, val_dataloader, train_num, val_num = build_dataloader(
            self.dataset_name, dataset_config, "train"
        )
        if dataset_config.x_dim is None:
            dataset_config.x_dim = train_dataloader.dataset.x_dim

        current_sequence_len = initial_sequence_len

        logger.info("Training samples: {}".format(train_num))
        logger.info("Validation samples: {}".format(val_num))

        # Blank val_criteria is recon,spectrum: each term divided by its
        # first validation epoch, then summed. Explicit recon, or
        # legacy_defaults / legacy_windowing, stays on the raw val_loss
        # comparison. The figure still records raw recon, spectrum, and
        # the Appendix B drift terms unless that legacy switch is on and
        # neither spectrum nor a drift term is selected. The windows, the
        # flip, and val_drift_horizon are fixed here and are not resized.
        self._val_criteria = resolve_val_criteria(self.cfg)
        self._val_reference = None
        self._selection_skipped = False
        self._any_selection_eligible = False
        self._val_spectrum_skipped = None
        self._val_fixed_recon = None
        self._contiguous_val = bool(
            getattr(val_dataloader.dataset, "val_stretches", None)
        )
        self._metric_reference = None
        self._monitor_spectrum = spectrum_monitoring_enabled(
            self._val_criteria,
            legacy=legacy_defaults_from_config(self.cfg),
        )
        self._checkpoint_every = parse_checkpoint_every_n_epochs(
            self.cfg.get("Training", "checkpoint_every_n_epochs", fallback="0")
        )
        self._spectrum_windows = None
        self._spectrum_tf_steps = None
        self._spectrum_observation = dataset_config.observation_process
        self._drift_horizon = None
        self._x_true_var = None
        self._window_info = None
        self._ftle_fit = (1, 1)
        self._val_rosenstein = False
        self._score_rosenstein = False
        self._needs_gt_lle = False
        self._lle_gt = None
        self._lle_gt_per_window = None
        self._lle_per_window = None
        self._ftle_per_window = None
        # Latest epoch's per-timestep traces, kept only until the
        # checkpoint figure is drawn. Not written into the history.
        self._latest_drift_steps = None
        self._latest_drift_steps_varnorm = None
        self._embed_tau, self._embed_dim = _get_delay_params(self.dataset_name)
        if self._monitor_spectrum:
            n_windows = parse_optional_int(
                self.cfg.get("Training", "val_spectrum_n_windows", fallback=None),
                10,
                "val_spectrum_n_windows",
            )
            window_len = parse_optional_int(
                self.cfg.get("Training", "val_spectrum_window_len", fallback=None),
                2000,
                "val_spectrum_window_len",
            )
            tf_steps = parse_optional_int(
                self.cfg.get("Training", "val_spectrum_tf_steps", fallback=None),
                1000,
                "val_spectrum_tf_steps",
            )
            if tf_steps < 0 or tf_steps >= window_len:
                raise ValueError(
                    "val_spectrum_tf_steps must be in [0, val_spectrum_window_len), "
                    f"got tf_steps={tf_steps}, window_len={window_len}."
                )
            self._spectrum_tf_steps = tf_steps
            self._drift_horizon = parse_val_drift_horizon(
                self.cfg.get("Training", "val_drift_horizon", fallback=None)
            )
            enforce_drift_span(window_len, tf_steps, self._drift_horizon)
            min_observed = parse_unit_interval(
                self.cfg.get("Training", "val_window_tf_min_observed", fallback=None),
                DEFAULT_TF_MIN_OBSERVED,
                "val_window_tf_min_observed",
            )
            max_gap_frac = parse_unit_interval(
                self.cfg.get("Training", "val_window_tf_max_gap_frac", fallback=None),
                DEFAULT_TF_MAX_GAP_FRAC,
                "val_window_tf_max_gap_frac",
            )
            self._ftle_fit = resolve_ftle_fit(
                self.cfg.get("Training", "val_ftle_fit_start", fallback=None),
                self.cfg.get("Training", "val_ftle_fit_end", fallback=None),
                self._drift_horizon,
            )
            self._val_rosenstein = parse_val_rosenstein(
                self.cfg.get("Training", "val_rosenstein", fallback=None)
            )
            # ftle and rosenstein_lle are |estimate - GT LLE|, so the
            # ground-truth exponent is required even when the Rosenstein
            # switch is off. Selecting rosenstein_lle also scores it
            # every epoch.
            self._score_rosenstein = self._val_rosenstein or (
                "rosenstein_lle" in self._val_criteria
            )
            self._needs_gt_lle = self._score_rosenstein or (
                "ftle" in self._val_criteria
            )
            if self._contiguous_val:
                max_gap_steps = parse_optional_int(
                    self.cfg.get("Training", "val_window_max_gap_steps", fallback=None),
                    10,
                    "val_window_max_gap_steps",
                )
                stride = parse_optional_int(
                    self.cfg.get("Training", "val_window_stride", fallback=None),
                    max(1, window_len // 2),
                    "val_window_stride",
                )
                if tf_steps < 1:
                    raise ValueError(
                        "contiguous_val_split needs val_spectrum_tf_steps >= 1."
                    )
                self._spectrum_windows, self._window_info = (
                    capture_contiguous_val_windows(
                        val_dataloader.dataset,
                        n_windows=n_windows,
                        window_len=window_len,
                        tf_steps=tf_steps,
                        min_observed=min_observed,
                        max_gap_steps=max_gap_steps,
                        stride=stride,
                    )
                )
                cinfo = self._window_info
                logger.info(
                    "Contiguous validation windows: {} of {} requested, from {} "
                    "candidate(s) at stride {} ({} pass both halves); stretches "
                    "{}; segment ids {}. Fixed-window recon, spectrum, drift and "
                    "Lyapunov all use this set every epoch; recon selection uses "
                    "the fixed-window recon.".format(
                        cinfo["n_used"],
                        cinfo["n_requested"],
                        cinfo["n_candidates"],
                        cinfo["stride"],
                        cinfo["n_passing"],
                        cinfo["stretches"],
                        cinfo["segment_ids"].tolist(),
                    )
                )
                if cinfo["n_used"] < n_windows:
                    logger.warning(
                        "Only {} validation window(s) pass both halves "
                        "(asked for {}). Using all of them; no failing window "
                        "is used.".format(cinfo["n_used"], n_windows)
                    )
            else:
                self._spectrum_windows, self._window_info = capture_fixed_spectrum_windows(
                    val_dataloader.dataset,
                    n_windows=n_windows,
                    window_len=window_len,
                    tf_steps=tf_steps,
                    min_observed=min_observed,
                    max_gap_frac=max_gap_frac,
                    return_info=True,
                )
            for window in self._spectrum_windows:
                enforce_drift_span(
                    np.asarray(window).shape[0], tf_steps, self._drift_horizon
                )
            self._x_true_var = x_true_variance(
                self._spectrum_windows, self._spectrum_observation
            )
            logger.info(
                "Spectrum validation: {} fixed window(s) of length {}, "
                "teacher-forced {}, free-run the rest. "
                "These copies are not passed to update_sequence_length.".format(
                    len(self._spectrum_windows), window_len, tf_steps
                )
            )
            logger.info(
                "Drift validation fixed at start: flip {}, horizon {}, "
                "variance over {} channel(s). Finite-time fit k={}..{}.".format(
                    tf_steps,
                    self._drift_horizon,
                    int(self._x_true_var.shape[0]),
                    self._ftle_fit[0],
                    self._ftle_fit[1],
                )
            )
            info = self._window_info
            logger.info(
                "Validation windows fixed at start (seed {}): starts {}, "
                "TF observed {}, max missing run {}, qualified {}.".format(
                    info["seed"],
                    info["starts"].tolist(),
                    [round(float(v), 4) for v in info["tf_observed"]],
                    info["tf_max_gap"].tolist(),
                    info["qualified"].tolist(),
                )
            )
            if info["relaxed"]:
                logger.warning(
                    "Fewer than {} validation windows met TF observed "
                    "fraction >= {} and max gap fraction {}. Using the "
                    "best-ranked windows listed above.".format(
                        n_windows, min_observed, max_gap_frac
                    )
                )
            if self._score_rosenstein and not self._val_rosenstein:
                logger.info(
                    "rosenstein_lle is selected, so Rosenstein is scored "
                    "every epoch even though val_rosenstein is false."
                )
            elif self._needs_gt_lle and not self._val_rosenstein:
                logger.info(
                    "ftle is selected, so the ground-truth Rosenstein "
                    "exponent is computed once even though val_rosenstein "
                    "is false."
                )
            if any(
                name in ("error_drift_cross", "error_drift_cross_varnorm")
                for name in self._val_criteria
            ):
                logger.info(
                    "Cross-term criteria use |cross|. The plotted series "
                    "stays signed."
                )
            if any(name in ("ftle", "rosenstein_lle") for name in self._val_criteria):
                logger.info("Lyapunov criteria use |estimate - GT LLE|.")
            if self._needs_gt_lle:
                started = time.perf_counter()
                self._lle_gt, _gt_std, self._lle_gt_per_window = rosenstein_on_free_run(
                    self._spectrum_windows,
                    tf_steps,
                    m=self._embed_dim,
                    tau=self._embed_tau,
                    observation_process=self._spectrum_observation,
                )
                elapsed = time.perf_counter() - started
                if self._score_rosenstein:
                    logger.info(
                        "Rosenstein GT LLE {:.6f} on the free-run half "
                        "(embed m={}, tau={}) in {:.3f}s. The same cost is "
                        "paid again each epoch on the autonomous pass.".format(
                            self._lle_gt, self._embed_dim, self._embed_tau, elapsed
                        )
                    )
                else:
                    logger.info(
                        "Rosenstein GT LLE {:.6f} on the free-run half "
                        "(embed m={}, tau={}) in {:.3f}s. Per-epoch "
                        "Rosenstein stays off.".format(
                            self._lle_gt, self._embed_dim, self._embed_tau, elapsed
                        )
                    )
            else:
                logger.info(
                    "val_rosenstein is false: skipping Rosenstein LLE. "
                    "The finite-time exponent is still computed."
                )

        ######################
        ### Batch Training ###
        ######################

        # Load training parameters
        epochs = self.cfg.getint("Training", "epochs")
        early_stop_patience = self.cfg.getint("Training", "early_stop_patience")
        save_frequency = self.cfg.getint("Training", "save_frequency")
        beta = self.cfg.getfloat("Training", "beta")

        # Create python list for loss
        loss_log = None
        sigmas_history = None

        def new_sigmas_history():
            """One row per entry of the shared sigmas parameter."""
            current = self.model.sigmas.detach().cpu().numpy().reshape(-1)
            history = np.zeros((current.shape[0], epochs))
            history[:, 0] = current
            return history

        if not self.params["reload"]:
            train_loss = np.zeros((epochs,))
            val_loss = np.zeros((epochs,))
            train_recon = np.zeros((epochs,))
            train_kl = np.zeros((epochs,))
            val_recon = np.zeros((epochs,))
            val_kl = np.zeros((epochs,))
            best_val_loss = np.inf
            cpt_patience = 0
            cur_best_epoch = epochs
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            start_epoch = -1
            if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
                sigmas_history = new_sigmas_history()
        else:
            cp_file = os.path.join(save_dir, "{}_checkpoint.pt".format(self.model_name))
            try:
                checkpoint = torch.load(cp_file, weights_only=False)
            except TypeError:
                checkpoint = torch.load(cp_file)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optim_state_dict"])
            start_epoch = checkpoint["epoch"]
            loss_log = checkpoint["loss_log"]
            if not isinstance(loss_log, dict):
                loss_log = {}
            logger.info("Resuming trainning: epoch: {}".format(start_epoch))
            train_loss = series_or_new(loss_log, "train_loss", epochs)
            val_loss = series_or_new(loss_log, "val_loss", epochs)
            train_recon = series_or_new(loss_log, "train_recon", epochs)
            train_kl = series_or_new(loss_log, "train_kl", epochs)
            val_recon = series_or_new(loss_log, "val_recon", epochs)
            val_kl = series_or_new(loss_log, "val_kl", epochs)
            best_val_loss = checkpoint["best_val_loss"]
            # A finite best from before the resume means selection already
            # ran; do not let the no-finite-spectrum fallback overwrite it.
            self._any_selection_eligible = bool(np.isfinite(best_val_loss))
            cpt_patience = 0
            cur_best_epoch = start_epoch
            best_state_dict = self.model.state_dict()
            best_optim_dict = optimizer.state_dict()
            if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
                sigmas_history = new_sigmas_history()

        # PLRNN / shPLRNN diagonal A. Independent of MT mixing sigmas.
        a_sigmas_history = init_A_sigmas_history(self.model, epochs)
        if isinstance(loss_log, dict):
            if sigmas_history is not None and "sigmas_history" in loss_log:
                sigmas_history = restore_matrix(
                    loss_log["sigmas_history"], sigmas_history.shape
                )
            if a_sigmas_history is not None and "A_sigmas_history" in loss_log:
                a_sigmas_history = restore_matrix(
                    loss_log["A_sigmas_history"], a_sigmas_history.shape
                )

        current_auto_warm = (
            self.auto_warm_start
            if self.enable_autonomous_warmup
            else self.auto_warm_limit
        )
        auto_warm_values = series_or_new(
            loss_log, "auto_warm_values", epochs, fill="hold"
        )

        # if model is vrnn or mt_vrnn, then kl_warm is used
        if self.enable_kl_warmup:
            current_kl_warm = 0
        else:
            current_kl_warm = 1

        # noise config variables
        self.noise_sampling_method = self.cfg.get(
            "Training", "noise_sampling_method", fallback="none"
        )
        self.noise_std_factor = self.cfg.getfloat(
            "Training", "noise_std_factor", fallback=0.1
        )
        self.noise_target = self.cfg.get("Training", "noise_target", fallback="both")
        current_noise_warm = (
            self.noise_warm_start if self.enable_noise_warmup else self.noise_warm_limit
        )
        noise_warm_values = series_or_new(
            loss_log, "noise_warm_values", epochs, fill="hold"
        )

        # stores the value of kl_warm at the epoch
        kl_warm_values = series_or_new(
            loss_log, "kl_warm_values", epochs, fill="hold"
        )
        best_state_epochs = series_or_new(
            loss_log, "best_state_epochs", epochs, fill="hold"
        )
        cpt_patience_epochs = series_or_new(loss_log, "cpt_patience_epochs", epochs)
        patience_threshold_epochs = series_or_new(
            loss_log, "patience_threshold_epochs", epochs, fill="hold"
        )
        delta_per_epoch = series_or_new(loss_log, "delta_per_epoch", epochs)
        if isinstance(loss_log, dict) and "sequence_len_values" in loss_log:
            sequence_len_values = series_or_new(
                loss_log, "sequence_len_values", epochs, fill="hold"
            )
        else:
            sequence_len_values = np.full((epochs,), initial_sequence_len)
        windows_kept = series_or_new(loss_log, "windows_kept", epochs, fill="hold")
        windows_dropped_nan = series_or_new(
            loss_log, "windows_dropped_nan", epochs, fill="hold"
        )
        windows_dropped_shift = series_or_new(
            loss_log, "windows_dropped_shift", epochs, fill="hold"
        )
        gap_split_counts = series_or_new(loss_log, "gap_splits", epochs, fill="hold")
        mean_start_shift = series_or_new(
            loss_log, "mean_start_shift", epochs, fill="hold"
        )
        delta_rolling_window_size = 6
        kl_delta_threshold = -1e-4
        auto_delta_threshold = 1e-4
        delta_threshold = 1e-4
        warm_up_happened = False
        val_spectrum = series_or_new(loss_log, "val_spectrum", epochs)
        val_criteria_score = series_or_new(loss_log, "val_criteria_score", epochs)
        self._val_spectrum_skipped = series_or_new(
            loss_log, "val_spectrum_skipped", epochs
        )
        if self._contiguous_val and self._spectrum_windows is not None:
            self._val_fixed_recon = series_or_new(loss_log, "val_fixed_recon", epochs)
        time_train = series_or_new(loss_log, "time_train", epochs)
        time_val = series_or_new(loss_log, "time_val", epochs)
        time_val_forward = series_or_new(loss_log, "time_val_forward", epochs)
        time_spectrum = series_or_new(loss_log, "time_spectrum", epochs)
        time_drift = series_or_new(loss_log, "time_drift", epochs)
        time_rosenstein = series_or_new(loss_log, "time_rosenstein", epochs)
        time_plot = series_or_new(loss_log, "time_plot", epochs)
        time_eval = series_or_new(loss_log, "time_eval", epochs)
        train_subband = series_or_new(loss_log, "train_subband", epochs)
        val_subband = series_or_new(loss_log, "val_subband", epochs)
        subband_warm_values = series_or_new(
            loss_log, "subband_warm_values", epochs, fill="hold"
        )
        subband_loss_weight_values = series_or_new(
            loss_log, "subband_loss_weight_values", epochs, fill="hold"
        )
        train_decorr = series_or_new(loss_log, "train_decorr", epochs)
        val_decorr = series_or_new(loss_log, "val_decorr", epochs)
        decorr_warm_values = series_or_new(
            loss_log, "decorr_warm_values", epochs, fill="hold"
        )
        decorr_loss_weight_values = series_or_new(
            loss_log, "decorr_loss_weight_values", epochs, fill="hold"
        )
        bump_events = []
        if isinstance(loss_log, dict) and isinstance(loss_log.get("bump_events"), list):
            bump_events = [dict(row) for row in loss_log["bump_events"]]
        drift_source = drift_series_source(loss_log)
        drift_arrays = {
            "val_" + name: series_or_new(drift_source, "val_" + name, epochs)
            for name in DRIFT_METRIC_NAMES
        }
        n_val_windows = (
            0 if self._spectrum_windows is None else len(self._spectrum_windows)
        )
        if self._monitor_spectrum:
            self._lle_per_window = np.full((epochs, n_val_windows), np.nan)
            self._ftle_per_window = np.full((epochs, n_val_windows), np.nan)
        if self.params["reload"] and isinstance(loss_log, dict):
            saved_ref = loss_log.get("val_criteria_reference")
            if isinstance(saved_ref, dict):
                aliased = {}
                for name, value in saved_ref.items():
                    key = canonical_criterion_name(name)
                    if key not in aliased or str(name) == key:
                        aliased[key] = float(value)
                restored = {}
                for name in self._val_criteria:
                    if name in aliased:
                        restored[name] = aliased[name]
                if len(restored) == len(self._val_criteria):
                    self._val_reference = restored
            saved_metrics = loss_log.get("val_metric_reference")
            if isinstance(saved_metrics, dict) and saved_metrics:
                restored_metrics = {}
                for name, value in saved_metrics.items():
                    key = canonical_criterion_name(name)
                    if key not in restored_metrics or str(name) == key:
                        restored_metrics[key] = float(value)
                self._metric_reference = restored_metrics
            if self._monitor_spectrum:
                saved_horizon = loss_log.get("val_drift_horizon")
                if saved_horizon is not None and int(saved_horizon) != int(
                    self._drift_horizon
                ):
                    raise ValueError(
                        "val_drift_horizon is fixed at training start "
                        f"({int(saved_horizon)}); this resume parsed "
                        f"{int(self._drift_horizon)}."
                    )
                saved_flip = loss_log.get("val_drift_flip")
                if saved_flip is not None and int(saved_flip) != int(
                    self._spectrum_tf_steps
                ):
                    raise ValueError(
                        "drift flip is fixed at training start "
                        f"({int(saved_flip)}); this resume parsed "
                        f"{int(self._spectrum_tf_steps)}."
                    )
                saved_var = loss_log.get("val_x_true_var")
                if saved_var is not None:
                    saved_var = np.asarray(saved_var, dtype=float).reshape(-1)
                    if saved_var.shape != self._x_true_var.shape or not np.allclose(
                        saved_var, self._x_true_var, equal_nan=True
                    ):
                        raise ValueError(
                            "val_x_true_var is fixed from the validation "
                            "windows at training start and does not match "
                            "this resume."
                        )
                saved_starts = loss_log.get("val_window_starts")
                if saved_starts is not None and self._window_info is not None:
                    saved_starts = np.asarray(saved_starts, dtype=np.int64).reshape(-1)
                    current = np.asarray(self._window_info["starts"], dtype=np.int64)
                    if saved_starts.shape != current.shape or not np.array_equal(
                        saved_starts, current
                    ):
                        raise ValueError(
                            "validation window starts are fixed at training "
                            f"start ({saved_starts.tolist()}); this resume "
                            f"chose {current.tolist()}."
                        )
                saved_fit_start = loss_log.get("val_ftle_fit_start")
                saved_fit_end = loss_log.get("val_ftle_fit_end")
                if saved_fit_start is not None and int(saved_fit_start) != int(
                    self._ftle_fit[0]
                ):
                    raise ValueError(
                        "val_ftle_fit_start is fixed at training start "
                        f"({int(saved_fit_start)}); this resume parsed "
                        f"{int(self._ftle_fit[0])}."
                    )
                if saved_fit_end is not None and int(saved_fit_end) != int(
                    self._ftle_fit[1]
                ):
                    raise ValueError(
                        "val_ftle_fit_end is fixed at training start "
                        f"({int(saved_fit_end)}); this resume parsed "
                        f"{int(self._ftle_fit[1])}."
                    )
                if "val_lle_per_window" in loss_log:
                    self._lle_per_window = restore_per_window(
                        loss_log["val_lle_per_window"], epochs, n_val_windows
                    )
                if "val_ftle_per_window" in loss_log:
                    self._ftle_per_window = restore_per_window(
                        loss_log["val_ftle_per_window"], epochs, n_val_windows
                    )
                saved_gt_windows = loss_log.get("val_lle_gt_per_window")
                if saved_gt_windows is not None and self._lle_gt_per_window is not None:
                    saved_gt_windows = np.asarray(saved_gt_windows, dtype=float).reshape(-1)
                    if saved_gt_windows.shape != self._lle_gt_per_window.shape or not np.allclose(
                        saved_gt_windows, self._lle_gt_per_window, equal_nan=True
                    ):
                        raise ValueError(
                            "val_lle_gt_per_window is fixed from the "
                            "validation windows at training start and does "
                            "not match this resume."
                        )

        # Blank warmup_order is window_first. A blank warmup_patience_start
        # is max(1, round(frac * early_stop_patience)), frac 0.2.
        # Sequence-length bumps hold that value, then it rises over the
        # autonomous and KL bumps. An explicit warmup_patience_start
        # overrides the fraction. auto_first uses early_stop_patience
        # for every bump. legacy_defaults or legacy_windowing restores
        # auto_first and that single patience when those keys are blank.
        warmup_order, warmup_patience_start = warmup_settings_from_config(self.cfg)
        warmup = WarmupCurriculum(
            order=warmup_order,
            patience_start=warmup_patience_start,
            early_stop_patience=early_stop_patience,
            initial_sequence_len=initial_sequence_len,
            sequence_len=self.sequence_len,
            enable_autonomous=self.enable_autonomous_warmup,
            auto_start=self.auto_warm_start,
            auto_limit=self.auto_warm_limit,
            enable_noise=self.enable_noise_warmup,
            noise_start=self.noise_warm_start,
            noise_limit=self.noise_warm_limit,
            noise_reset_on_window=self.noise_warm_reset_on_window,
            enable_kl=self.enable_kl_warmup,
        )
        logger.info(
            "Warm-up order: {} (window steps {}, autonomous steps {}, "
            "KL steps {}, planned bumps {}). Patience start: {}. "
            "Early stop patience: {}.".format(
                warmup.order,
                warmup.window_steps,
                warmup.autonomous_steps,
                warmup.kl_steps,
                warmup.n_planned,
                (
                    "off"
                    if warmup.patience_start is None
                    else format_patience(warmup.patience_start)
                ),
                early_stop_patience,
            )
        )
        auto_kl_done = 0
        self._regularizers = parse_regularizer_settings(
            self.cfg,
            y_dim=int(self.model.y_dim),
            hidden_block=getattr(self.model, "hidden_block", None),
        )
        dataset_stats = snapshot_window_stats(train_dataloader.dataset)

        def history_at(end):
            logged = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_recon": train_recon,
                "train_kl": train_kl,
                "val_recon": val_recon,
                "val_kl": val_kl,
                "val_spectrum": val_spectrum,
                "val_criteria_score": val_criteria_score,
            }
            if self._monitor_spectrum:
                logged.update(drift_arrays)
                logged["val_spectrum_skipped"] = self._val_spectrum_skipped
                if self._val_fixed_recon is not None:
                    logged["val_fixed_recon"] = self._val_fixed_recon
            return build_training_history(
                end=end,
                model_name=self.model_name,
                tag=tag,
                auto_sampling_method=self.autonomous_sampling_method,
                noise_sampling_method=self.noise_sampling_method,
                delta_threshold=delta_threshold,
                val_criteria=self._val_criteria,
                val_reference=self._val_reference,
                metric_reference=self._metric_reference,
                spectrum_monitored=self._monitor_spectrum,
                drift_monitored=self._monitor_spectrum,
                drift_horizon=self._drift_horizon,
                drift_flip=self._spectrum_tf_steps,
                x_true_var=self._x_true_var,
                window_info=self._window_info,
                lle_gt=self._lle_gt,
                rosenstein=self._val_rosenstein if self._monitor_spectrum else None,
                ftle_fit=self._ftle_fit if self._monitor_spectrum else None,
                lle_per_window=self._lle_per_window,
                ftle_per_window=self._ftle_per_window,
                lle_gt_per_window=self._lle_gt_per_window,
                bump_events=bump_events,
                series={
                    **logged,
                    "kl_warm_values": kl_warm_values,
                    "auto_warm_values": auto_warm_values,
                    "noise_warm_values": noise_warm_values,
                    "sequence_len_values": sequence_len_values,
                    "cpt_patience_epochs": cpt_patience_epochs,
                    "patience_threshold_epochs": patience_threshold_epochs,
                    "delta_per_epoch": delta_per_epoch,
                    "best_state_epochs": best_state_epochs,
                    "windows_kept": windows_kept,
                    "windows_dropped_nan": windows_dropped_nan,
                    "windows_dropped_shift": windows_dropped_shift,
                    "gap_splits": gap_split_counts,
                    "mean_start_shift": mean_start_shift,
                    "time_train": time_train,
                    "time_val": time_val,
                    "time_val_forward": time_val_forward,
                    "time_spectrum": time_spectrum,
                    "time_drift": time_drift,
                    "time_rosenstein": time_rosenstein,
                    "time_plot": time_plot,
                    "time_eval": time_eval,
                    "train_subband": train_subband,
                    "val_subband": val_subband,
                    "subband_warm_values": subband_warm_values,
                    "subband_loss_weight_values": subband_loss_weight_values,
                    "train_decorr": train_decorr,
                    "val_decorr": val_decorr,
                    "decorr_warm_values": decorr_warm_values,
                    "decorr_loss_weight_values": decorr_loss_weight_values,
                },
                subband_enabled=self._regularizers.subband_on,
                decorr_enabled=self._regularizers.decorr_on,
                dataset_stats=dataset_stats,
                sigmas_history=(
                    sigmas_history
                    if self.model_name in ("MT_RNN", "MT_VRNN")
                    else None
                ),
                a_sigmas_history=a_sigmas_history,
            )

        def store_metric_times(epoch_index):
            metric_times = getattr(self, "_epoch_metric_times", {})
            time_val_forward[epoch_index] = float(metric_times.get("time_val_forward", 0.0))
            time_spectrum[epoch_index] = float(metric_times.get("time_spectrum", 0.0))
            time_drift[epoch_index] = float(metric_times.get("time_drift", 0.0))
            time_rosenstein[epoch_index] = float(metric_times.get("time_rosenstein", 0.0))

        def log_epoch_wall(epoch_index):
            store_metric_times(epoch_index)
            logger.info(
                format_timing_line(
                    epoch_index,
                    {
                        "time_train": time_train[epoch_index],
                        "time_val": time_val[epoch_index],
                        "time_val_forward": time_val_forward[epoch_index],
                        "time_spectrum": time_spectrum[epoch_index],
                        "time_drift": time_drift[epoch_index],
                        "time_rosenstein": time_rosenstein[epoch_index],
                        "time_plot": time_plot[epoch_index],
                        "time_eval": time_eval[epoch_index],
                    },
                )
            )

        self.model.to(self.device)

        # Train with mini-batch SGD
        ran_epoch = False
        stopped_early = False
        for epoch in range(start_epoch + 1, epochs):
            ran_epoch = True
            # The bump that ends this epoch increments auto_kl_done afterwards,
            # so the loss of the trigger epoch still uses the previous scale.
            # A disabled term stores a zero scale and a zero weight. The
            # warm-up panel hides those series.
            sub_scale = (
                warmup_progress(
                    auto_kl_done,
                    warmup.autonomous_steps,
                    warmup.kl_steps,
                    self._regularizers.subband_warmup,
                )
                if self._regularizers.subband_on
                else 0.0
            )
            dec_scale = (
                warmup_progress(
                    auto_kl_done,
                    warmup.autonomous_steps,
                    warmup.kl_steps,
                    self._regularizers.decorr_warmup,
                )
                if self._regularizers.decorr_on
                else 0.0
            )
            sub_w = float(self._regularizers.subband_weight) * sub_scale
            dec_w = float(self._regularizers.decorr_weight) * dec_scale
            subband_warm_values[epoch] = sub_scale
            decorr_warm_values[epoch] = dec_scale
            subband_loss_weight_values[epoch] = sub_w
            decorr_loss_weight_values[epoch] = dec_w

            start_time = datetime.datetime.now()

            # Define a mapping from sampling_method to corresponding parameters
            autonomous_configs = {
                "ss": {
                    "mode": "bernoulli_sampling",
                    "autonomous_ratio": lambda: current_auto_warm,
                },
                "ptf": {
                    "mode": "bernoulli_sampling",
                    "autonomous_ratio": lambda: self.sampling_ratio,
                },
                "mtf": {
                    "mode": "mix_sampling",
                    "autonomous_ratio": lambda: self.sampling_ratio,
                },
                "sm": {
                    "mode": "mix_sampling",
                    "autonomous_ratio": lambda: current_auto_warm,
                },
                "even_bursts": {
                    "mode": "even_bursts",
                    "autonomous_ratio": lambda: self.sampling_ratio
                    * current_sequence_len
                    / 10,
                },
            }

            # New: parallel structure just for noise
            if self.tie_noise_to_auto and self.enable_noise_warmup:
                current_noise_warm = (
                    current_auto_warm  # Direct tie for confounding-free baselines
                )

            noise_configs = {
                "none": None,  # disable noise entirely
                "ss": {  # scheduled probabilistic noise (recommended default / most analogous to SS)
                    "mode": "bernoulli_sampling",
                    "ratio_fn": lambda: current_noise_warm,  # anneal from low→high
                },
                "sm": {  # scheduled continuous noise (soft additive noise that ramps up)
                    "mode": "mix_sampling",
                    "ratio_fn": lambda: current_noise_warm,
                },
                "ptf": {  # fixed high-probability noise (like permanent dropout/jitter)
                    "mode": "bernoulli_sampling",
                    "ratio_fn": lambda: self.noise_ratio,  # fixed, e.g. 0.3–0.5
                },
                "mtf": {  # fixed continuous noise strength (always on, scaled fraction)
                    "mode": "mix_sampling",
                    "ratio_fn": lambda: self.noise_ratio,
                },
                # Optional extras if you want them later
                "even_bursts": {
                    "mode": "even_bursts",
                    "ratio_fn": lambda: self.noise_ratio * current_sequence_len / 10,
                },
            }

            autonomous_config = autonomous_configs[self.autonomous_sampling_method]
            noise_config = noise_configs.get(self.noise_sampling_method)

            torch.autograd.set_detect_anomaly(True)
            # Batch training
            sync_cuda(self.device)
            train_t0 = time.perf_counter()
            for _, batch_data in enumerate(train_dataloader):
                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)
                if batch_size != self.batch_size:
                    logger.warning(
                        f"[Learning Algo][epoch{epoch}][train] batch_size: {batch_size}, expected: {self.batch_size}"
                    )

                base_mode_selector = create_autonomous_mode_selector(
                    current_sequence_len,
                    mode=autonomous_config["mode"],
                    autonomous_ratio=autonomous_config["autonomous_ratio"](),
                    batch_size=batch_size,
                    x_dim=dataset_config.x_dim,
                    device=self.device,
                )

                # if nan is present in the batch data, overlay the mask to fill with autonomous mode
                if batch_data.isnan().any():
                    # ratio of nan
                    nan_ratio = batch_data.isnan().sum() / batch_data.numel()
                    logger.warning(
                        f"[Learning Algo][epoch{epoch}][train] NaN detected in training data: {nan_ratio*100}%. Will overlay on mask"
                    )
                    # replace where nan is present with 1 to fill with autonomous mode
                    model_mode_selector = torch.where(
                        batch_data.permute(1, 0, 2).isnan().detach(),
                        torch.tensor(1.0, device=self.device),
                        base_mode_selector,
                    )
                else:
                    model_mode_selector = base_mode_selector

                # [Masking Baseline] For observation_process == "only_x_indicate", force 100% TF on mask dimension
                if self._is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ):
                    model_mode_selector = model_mode_selector.clone()
                    model_mode_selector[:, :, 1::2] = 0.0  # Force pure TF on mask channel(s)

                # START: Compute masked batch std for adaptive noise #####################
                noise_config = noise_configs.get(self.noise_sampling_method)
                noise_selector = None
                if noise_config is not None:
                    current_noise_ratio = noise_config["ratio_fn"]()
                    noise_selector = create_autonomous_mode_selector(
                        current_sequence_len,
                        mode=noise_config["mode"],
                        autonomous_ratio=current_noise_ratio,
                        batch_size=batch_size,
                        x_dim=dataset_config.x_dim,
                        device=self.device,
                    )

                    # Restrict to observed + targeting (keep this)
                    # observed_mask = (
                    #     ~batch_data.permute(1, 0, 2).isnan().detach()
                    # )  # For selector restrict
                    # noise_selector *= observed_mask.float()

                    if self.noise_target == "tf_only":
                        noise_selector *= 1 - model_mode_selector
                    elif self.noise_target in ["ar_only", "hybrid_ar"]:
                        noise_selector *= model_mode_selector

                # END: Compute masked batch std for adaptive noise #####################
                print(
                    f"[Learning Algo][epoch{epoch}][train] sent to device: {self.device}"
                )

                ################################
                # Forward pass for training #
                ################################
                if (
                    self.dataset_name == "Lorenz63"
                    or self.dataset_name == "Sinusoid"
                    or self.dataset_name == "Xhro"
                    or self.dataset_name == "XhroPacketLoss"
                    or self.dataset_name == "XhroProper"
                    or self.dataset_name == "PhysioNet2012"
                    or self.dataset_name == "SHO"
                    or self.dataset_name == "DampedSHO"
                ):
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    recon_batch_data = self.model(
                        batch_data,
                        mode_selector=model_mode_selector,
                        noise_selector=noise_selector,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][train]",
                    )

                    # Now, prepare masked data for loss
                    is_observed = (
                        ~batch_data.isnan().detach()
                    ).float()  # 1 where GT available

                    if self.loss_mask_mode == "none":
                        loss_mask = (
                            is_observed  # Loss on all observed, ignore input mode
                        )
                    elif self.loss_mask_mode == "strict":
                        pure_tf = (
                            base_mode_selector == 0
                        ).float()  # 1 only if base was pure TF
                        loss_mask = (
                            is_observed * pure_tf
                        )  # Exclude scheduled autonomous/mixed
                    elif self.loss_mask_mode == "weighted":
                        tf_weight = (1.0 - base_mode_selector).clamp(
                            0.0, 1.0
                        )  # TF ratio, for interpolation
                        loss_mask = (
                            is_observed * tf_weight
                        )  # Weight loss by TF contribution
                    else:
                        raise ValueError(
                            f"Invalid loss_mask_mode: {self.loss_mask_mode}"
                        )

                    # Compute loss (using masking baseline if applicable)
                    if self._is_indicate_observation(
                        getattr(dataset_config, "observation_process", None)
                    ):
                        # [Masking Baseline] Loss computed only on signal dimension (ch 0), only on observed timesteps
                        loss_recon = self._prepare_masked_loss(
                            batch_data,
                            recon_batch_data,
                            loss_mask,
                            model_mode_selector,
                            dataset_config,
                        )
                    else:
                        # Standard loss computation (all dimensions, based on loss_mask_mode)
                        batch_data_masked_for_loss_func = batch_data * loss_mask
                        batch_data_masked_for_loss_func = torch.nan_to_num(
                            batch_data_masked_for_loss_func, nan=0.0
                        )
                        recon_batch_data_masked_for_loss_func = (
                            recon_batch_data * loss_mask
                        )
                        # Sanity check for NaNs (shouldn't happen after nan_to_num)
                        if batch_data_masked_for_loss_func.isnan().any():
                            logger.warning(
                                f"[Learning Algo][epoch{epoch}][train] NaN detected in batch_data_masked_for_loss_func!! This shouldn't happen."
                            )

                        loss_recon = loss_MSE(
                            batch_data_masked_for_loss_func,
                            recon_batch_data_masked_for_loss_func,
                        )

                    # Raise warning if nan is detected in recon loss
                    assert not torch.isnan(
                        loss_recon
                    ).any(), f"[Learning Algo][epoch{epoch}][train] Nan detected in recon loss"
                else:
                    logger.error("Unknown datset")
                seq_len, bs, _ = self.model.y.shape  # Sequence Length and Batch Size
                # Average Reconstruction Loss
                loss_recon_avg = loss_recon / (seq_len * bs)

                if self.model_name == "DSAE":
                    loss_kl_z = loss_KLD(
                        self.model.z_mean,
                        self.model.z_logvar,
                        self.model.z_mean_p,
                        self.model.z_logvar_p,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][train]",
                    )
                    loss_kl_v = loss_KLD(
                        self.model.v_mean,
                        self.model.v_logvar,
                        self.model.v_mean_p,
                        self.model.v_logvar_p,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][train]",
                    )
                    loss_kl = loss_kl_z + loss_kl_v
                elif self.model_name == "RNN" or self.model_name == "MT_RNN":
                    loss_kl = torch.zeros(1).to(self.device)
                else:
                    loss_kl = loss_KLD(
                        self.model.z_mean,
                        self.model.z_logvar,
                        self.model.z_mean_p,
                        self.model.z_logvar_p,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][train]",
                    )

                loss_kl_avg = (
                    current_kl_warm * beta * loss_kl / (seq_len * bs)
                )  # Average KL Divergence

                # Print device of loss
                print(
                    f"[Learning Algo][epoch{epoch}] loss_recon.device: {loss_recon.device}, loss_kl.device: {loss_kl.device}"
                )
                loss_tot_avg = loss_recon_avg + loss_kl_avg
                sub_loss, dec_loss = regularizer_losses(
                    self.model, batch_data, self._regularizers
                )
                loss_backward = loss_tot_avg + sub_w * sub_loss + dec_w * dec_loss
                optimizer.zero_grad()
                loss_backward.backward()

                # temporarily store detatched parameters
                current_named_params_grad = [
                    (name, param.grad.detach().cpu().numpy().copy())
                    for name, param in self.model.named_parameters()
                ]
                current_named_params = [
                    (name, param.detach().cpu().numpy().copy())
                    for name, param in self.model.named_parameters()
                ]

                parameter_norm = get_norm_from_named_params(current_named_params)
                grad_norm = get_norm_from_named_params(current_named_params_grad)
                logger.info(
                    f"[Learning Algo][epoch{epoch}] Parameter norm: {parameter_norm}, Gradient norm: {grad_norm}"
                )

                clipped_named_params_grad = None
                # Gradient clipping
                # check if model has model.type_rnn, if True, and if type_rnn=='RNN', then clip the gradient

                self.gradient_clip = self.cfg.getfloat(
                    "Training", "gradient_clip", fallback=0.0
                )
                if self.gradient_clip > 0.0:
                    # clip the gradient
                    logger.info(
                        f"[Learning Algo][epoch{epoch}] Gradient clipping: {self.gradient_clip}"
                    )
                    # Actually perform gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                    # store the clipped gradients
                    clipped_named_params_grad = [
                        (name, param.grad.detach().cpu().numpy().copy())
                        for name, param in self.model.named_parameters()
                    ]

                    clipped_grad_norm = get_norm_from_named_params(
                        clipped_named_params_grad
                    )
                    logger.info(
                        f"[Learning Algo][epoch{epoch}] Gradient norm clipped from {grad_norm} to {clipped_grad_norm}, parameter norm: {parameter_norm}"
                    )

                optimizer.step()
                # profiler.step()

                # tot is the tensor that was backward'd: recon + KL plus the
                # weighted subband and decorr terms. recon and KL stay separate.
                train_loss[epoch] += loss_backward.item() * bs
                train_recon[epoch] += loss_recon_avg.item() * bs
                train_kl[epoch] += loss_kl_avg.item() * bs
                train_subband[epoch] += float(sub_loss.detach()) * bs
                train_decorr[epoch] += float(dec_loss.detach()) * bs
                if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
                    sigmas_history[:, epoch] = (
                        self.model.sigmas.detach().cpu().numpy()
                    )
                update_A_sigmas_history(a_sigmas_history, self.model, epoch)

            sync_cuda(self.device)
            time_train[epoch] = time.perf_counter() - train_t0

            # Validation
            sync_cuda(self.device)
            val_t0 = time.perf_counter()
            for _, batch_data in enumerate(val_dataloader):

                batch_data = batch_data.to(self.device)
                batch_size = batch_data.size(0)

                base_mode_selector = create_autonomous_mode_selector(
                    current_sequence_len,
                    mode=autonomous_config["mode"],
                    autonomous_ratio=autonomous_config["autonomous_ratio"](),
                    batch_size=batch_size,
                    x_dim=dataset_config.x_dim,
                    device=self.device,
                )

                # if nan is present in the batch data, overlay the mask to fill with autonomous mode
                if batch_data.isnan().any():
                    # ratio of nan
                    nan_ratio = batch_data.isnan().sum() / batch_data.numel()
                    logger.warning(
                        f"[Learning Algo][epoch{epoch}][train] Nan detected in validation data: {nan_ratio}. Will overlay on mask"
                    )
                    # replace where nan is present with 1 to fill with autonomous mode
                    model_mode_selector = torch.where(
                        batch_data.permute(1, 0, 2).isnan().detach(),
                        torch.tensor(1.0, device=self.device),
                        base_mode_selector,
                    )
                else:
                    model_mode_selector = base_mode_selector

                # [Masking Baseline] For observation_process == "only_x_indicate", force 100% TF on mask dimension
                if self._is_indicate_observation(
                    getattr(dataset_config, "observation_process", None)
                ):
                    model_mode_selector = model_mode_selector.clone()
                    model_mode_selector[:, :, 1::2] = 0.0  # Force pure TF on mask channel(s)

                # START: Compute masked batch std for adaptive noise #####################
                noise_config = noise_configs.get(self.noise_sampling_method)
                noise_selector = None
                if noise_config is not None:
                    current_noise_ratio = noise_config["ratio_fn"]()
                    noise_selector = create_autonomous_mode_selector(
                        current_sequence_len,
                        mode=noise_config["mode"],
                        autonomous_ratio=current_noise_ratio,
                        batch_size=batch_size,
                        x_dim=dataset_config.x_dim,
                        device=self.device,
                    )

                    # Restrict to observed + targeting (keep this)
                    # observed_mask = (
                    #     ~batch_data.permute(1, 0, 2).isnan().detach()
                    # )  # For selector restrict
                    # noise_selector *= observed_mask.float()

                    if self.noise_target == "tf_only":
                        noise_selector *= 1 - model_mode_selector
                    elif self.noise_target in ["ar_only", "hybrid_ar"]:
                        noise_selector *= model_mode_selector

                print(
                    f"[Learning Algo][epoch{epoch}][val] sent to device: {self.device}"
                )
                # END: Compute masked batch std for adaptive noise #####################

                ################################
                # Forward pass for validation #
                ################################
                if (
                    self.dataset_name == "Lorenz63"
                    or self.dataset_name == "Sinusoid"
                    or self.dataset_name == "Xhro"
                    or self.dataset_name == "XhroPacketLoss"
                    or self.dataset_name == "XhroProper"
                    or self.dataset_name == "PhysioNet2012"
                    or self.dataset_name == "SHO"
                    or self.dataset_name == "DampedSHO"
                ):
                    # Permute the batch data to match the expected input shape
                    # (batch_size, seq_len, x_dim) -> (seq_len, batch_size, x_dim)
                    batch_data = batch_data.permute(1, 0, 2)
                    # Run the model in validation mode
                    recon_batch_data = self.model(
                        batch_data,
                        mode_selector=model_mode_selector,
                        noise_selector=noise_selector,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][val]",
                    )

                    # Now, prepare masked data for loss
                    is_observed = (
                        ~batch_data.isnan().detach()
                    ).float()  # 1 where GT available

                    if self.loss_mask_mode == "none":
                        loss_mask = (
                            is_observed  # Loss on all observed, ignore input mode
                        )
                    elif self.loss_mask_mode == "strict":
                        pure_tf = (
                            base_mode_selector == 0
                        ).float()  # 1 only if base was pure TF
                        loss_mask = (
                            is_observed * pure_tf
                        )  # Exclude scheduled autonomous/mixed
                    elif self.loss_mask_mode == "weighted":
                        tf_weight = (1.0 - base_mode_selector).clamp(
                            0.0, 1.0
                        )  # TF ratio, for interpolation
                        loss_mask = (
                            is_observed * tf_weight
                        )  # Weight loss by TF contribution
                    else:
                        raise ValueError(
                            f"Invalid loss_mask_mode: {self.loss_mask_mode}"
                        )

                    # Compute loss (using masking baseline if applicable)
                    if self._is_indicate_observation(
                        getattr(dataset_config, "observation_process", None)
                    ):
                        # [Masking Baseline] Loss computed only on signal dimension (ch 0), only on observed timesteps
                        loss_recon = self._prepare_masked_loss(
                            batch_data,
                            recon_batch_data,
                            loss_mask,
                            model_mode_selector,
                            dataset_config,
                        )
                    else:
                        # Standard loss computation (all dimensions, based on loss_mask_mode)
                        batch_data_masked_for_loss_func = batch_data * loss_mask
                        batch_data_masked_for_loss_func = torch.nan_to_num(
                            batch_data_masked_for_loss_func, nan=0.0
                        )
                        recon_batch_data_masked_for_loss_func = (
                            recon_batch_data * loss_mask
                        )
                        # Sanity check for NaNs (shouldn't happen after nan_to_num)
                        if batch_data_masked_for_loss_func.isnan().any():
                            logger.warning(
                                f"[Learning Algo][epoch{epoch}][train] NaN detected in batch_data_masked_for_loss_func!! This shouldn't happen."
                            )

                        loss_recon = loss_MSE(
                            batch_data_masked_for_loss_func,
                            recon_batch_data_masked_for_loss_func,
                        )

                    # Raise warning if nan is detected in recon loss
                    assert not torch.isnan(
                        loss_recon
                    ).any(), (
                        f"[Learning Algo][epoch{epoch}][val] Nan detected in recon loss"
                    )

                seq_len, bs, _ = self.model.y.shape
                loss_recon_avg = loss_recon / (seq_len * bs)

                if self.model_name == "DSAE":
                    loss_kl_z = loss_KLD(
                        self.model.z_mean,
                        self.model.z_logvar,
                        self.model.z_mean_p,
                        self.model.z_logvar_p,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][val]",
                    )
                    loss_kl_v = loss_KLD(
                        self.model.v_mean,
                        self.model.v_logvar,
                        self.model.v_mean_p,
                        self.model.v_logvar_p,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][val]",
                    )
                    loss_kl = loss_kl_z + loss_kl_v
                elif self.model_name == "RNN" or self.model_name == "MT_RNN":
                    loss_kl = torch.zeros(1).to(self.device)
                else:
                    loss_kl = loss_KLD(
                        self.model.z_mean,
                        self.model.z_logvar,
                        self.model.z_mean_p,
                        self.model.z_logvar_p,
                        logger=logger,
                        from_instance=f"[Learning Algo][epoch{epoch}][val]",
                    )
                loss_kl_avg = current_kl_warm * beta * loss_kl / (seq_len * bs)

                print(
                    f"[Learning Algo][epoch{epoch}] loss_recon.device: {loss_recon.device}, loss_kl.device: {loss_kl.device}"
                )
                loss_tot_avg = loss_recon_avg + loss_kl_avg
                sub_loss, dec_loss = regularizer_losses(
                    self.model, batch_data, self._regularizers
                )

                val_loss[epoch] += loss_tot_avg.item() * bs
                val_recon[epoch] += loss_recon_avg.item() * bs
                val_kl[epoch] += loss_kl_avg.item() * bs
                val_subband[epoch] += float(sub_loss.detach()) * bs
                val_decorr[epoch] += float(dec_loss.detach()) * bs

            sync_cuda(self.device)
            time_val[epoch] = time.perf_counter() - val_t0

            # Loss normalization
            train_loss[epoch] = train_loss[epoch] / train_num
            train_recon[epoch] = train_recon[epoch] / train_num
            train_kl[epoch] = train_kl[epoch] / train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            val_recon[epoch] = val_recon[epoch] / val_num
            val_kl[epoch] = val_kl[epoch] / val_num
            train_subband[epoch] = train_subband[epoch] / train_num
            train_decorr[epoch] = train_decorr[epoch] / train_num
            val_subband[epoch] = val_subband[epoch] / val_num
            val_decorr[epoch] = val_decorr[epoch] / val_num

            # Training time
            end_time = datetime.datetime.now()
            interval = end_time - start_time
            logger.info(
                "Epoch: {} training time {}".format(epoch, str(interval).split(".")[0])
            )
            logger.info(
                "Train => tot: {:.4f} recon {:.4f} KL {:.4f} subband {:.4f} decorr {:.4f} "
                "Val => tot: {:.4f} recon {:.4f} KL {:.4f} subband {:.4f} decorr {:.4f}".format(
                    train_loss[epoch],
                    train_recon[epoch],
                    train_kl[epoch],
                    train_subband[epoch],
                    train_decorr[epoch],
                    val_loss[epoch],
                    val_recon[epoch],
                    val_kl[epoch],
                    val_subband[epoch],
                    val_decorr[epoch],
                )
            )
            if self._regularizers.subband_on:
                logger.info(
                    "Train => subband {:.4f} weight {:.4f} Val => subband {:.4f} weight {:.4f}".format(
                        train_subband[epoch],
                        sub_w,
                        val_subband[epoch],
                        sub_w,
                    )
                )
            if self._regularizers.decorr_on:
                logger.info(
                    "Train => decorr {:.4f} weight {:.4f} Val => decorr {:.4f} weight {:.4f}".format(
                        train_decorr[epoch],
                        dec_w,
                        val_decorr[epoch],
                        dec_w,
                    )
                )

            selection_value, selection_raw, selection_norm = self._epoch_selection_value(
                epoch, val_loss, val_spectrum, val_criteria_score, drift_arrays
            )
            store_metric_times(epoch)

            # Save the best model
            new_best = False
            selection_skipped = bool(self._selection_skipped)
            if warm_up_happened and not selection_skipped:
                logger.info("Warm-up happened, saving the best model")
                warm_up_happened = False
                if self._val_criteria == ("recon",):
                    best_val_loss = selection_recon_value(self, epoch, val_loss)
                else:
                    best_val_loss = selection_value
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
                new_best = True

            # identify the delta between current val_loss and best_val_loss
            # recon keeps the historical comparison exactly.
            if selection_skipped:
                delta = float("nan")
            elif self._val_criteria == ("recon",):
                delta = selection_recon_value(self, epoch, val_loss) - best_val_loss
            else:
                delta = selection_value - best_val_loss
            delta_per_epoch[epoch] = delta
            # calculate the rolling average of delta if the window size is reached
            if epoch >= delta_rolling_window_size:
                delta_rolling_average = np.mean(
                    delta_per_epoch[-delta_rolling_window_size:]
                )
            else:
                delta_rolling_average = np.mean(delta_per_epoch)

            # Save the best model, update patience and best_val_loss
            # A skipped epoch is never an improvement: patience still counts it.
            improved = (not selection_skipped) and delta < -delta_threshold
            if improved:
                cpt_patience = 0
                if self._val_criteria == ("recon",):
                    best_val_loss = selection_recon_value(self, epoch, val_loss)
                else:
                    best_val_loss = selection_value
                best_state_dict = self.model.state_dict()
                best_optim_dict = optimizer.state_dict()
                cur_best_epoch = epoch
                new_best = True
            else:
                cpt_patience += 1
                print(
                    f"[Learning Algo][epoch{epoch}] Updated cpt_patience: {cpt_patience}"
                )
                logger.info(
                    "Updated cpt_patience: {}, delta: {}".format(cpt_patience, delta)
                )
            logger.info(
                format_selection_line(
                    epoch,
                    selection_value,
                    selection_raw,
                    selection_norm,
                    improved,
                )
            )
            if new_best:
                logger.info(
                    format_new_best_line(
                        epoch, selection_value, selection_raw, selection_norm
                    )
                )

            # Store the warm-up values
            kl_warm_values[epoch] = current_kl_warm
            auto_warm_values[epoch] = current_auto_warm
            noise_warm_values[epoch] = current_noise_warm
            sequence_len_values[epoch] = current_sequence_len
            window_snap = snapshot_window_stats(train_dataloader.dataset)
            windows_kept[epoch] = window_snap["windows_kept"]
            windows_dropped_nan[epoch] = window_snap["windows_dropped_nan"]
            windows_dropped_shift[epoch] = window_snap["windows_dropped_shift"]
            gap_split_counts[epoch] = window_snap["gap_splits"]
            mean_start_shift[epoch] = window_snap["mean_start_shift"]
            dataset_stats = window_snap
            # Stop traning if early-stop triggers
            # Only stop when all enabled warm-ups are finished AND the sequence
            # length has already been increased to the requested maximum. This
            # allows the sequence-length warm-up to still occur when other
            # warm-ups are disabled.
            # auto_first: autonomous+noise, then KL, then sequence length.
            # window_first: sequence length first (autonomous fraction stays at
            # auto_warm_start), then autonomous+noise, then KL. Same 0.1 bumps.
            # window_first holds warmup_patience_start on sequence-length
            # bumps, then raises it only across autonomous and KL bumps.
            # auto_first, and a blank start, use early_stop_patience for
            # every bump. The final stop always uses early_stop_patience.
            warmups_finished = warmup.finished(
                current_auto_warm,
                current_noise_warm,
                current_kl_warm,
                current_sequence_len,
            )
            active_patience = warmup.active_patience(
                auto_kl_done,
                current_auto_warm,
                current_noise_warm,
                current_kl_warm,
                current_sequence_len,
            )
            logger.info(
                "Epoch: {} active patience: {}".format(
                    epoch, format_patience(active_patience)
                )
            )
            action = warmup_action(
                cpt_patience, active_patience, warmups_finished
            )
            patience_threshold_epochs[epoch] = active_patience
            if action == "stop":
                cpt_patience_epochs[epoch] = cpt_patience
                best_state_epochs[epoch] = cur_best_epoch
                logger.info("Early stop patience achieved")
                logger.info(
                    format_early_stop_line(epoch, cpt_patience, active_patience)
                )
                log_epoch_wall(epoch)
                stopped_early = True
                break
            if action == "bump":
                cpt_at_trigger = cpt_patience
                patience_before = float(active_patience)
                old_auto = float(current_auto_warm)
                old_noise = float(current_noise_warm)
                old_kl = float(current_kl_warm)
                old_sequence = float(current_sequence_len)
                cpt_patience = 0
                warm_up_happened = True
                bumped = warmup.apply_bump(
                    current_auto_warm,
                    current_noise_warm,
                    current_kl_warm,
                    current_sequence_len,
                )
                current_auto_warm = bumped.auto
                current_noise_warm = bumped.noise
                current_kl_warm = bumped.kl
                current_sequence_len = bumped.sequence_len
                if bumped.kind == "auto":
                    if bumped.auto_bumped:
                        logger.info("Bumping auto warm-up")
                        logger.info(
                            "Autonomous warm-up, anneal coeff: {}".format(
                                current_auto_warm
                            )
                        )
                    if bumped.noise_bumped:
                        logger.info("Bumping noise warm-up")
                        logger.info(
                            "Noise warm-up, noise coeff: {}".format(
                                current_noise_warm
                            )
                        )
                elif bumped.kind == "kl":
                    logger.info("Bumping KL warm-up")
                elif bumped.kind == "window":
                    logger.info("Bumping sequence length")
                    train_dataloader.dataset.update_sequence_length(
                        current_sequence_len,
                    )
                    val_dataloader.dataset.update_sequence_length(
                        current_sequence_len
                    )
                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataloader.dataset,
                        batch_size=train_dataloader.batch_size,
                        shuffle=self.shuffle,
                        num_workers=train_dataloader.num_workers,
                        pin_memory=True,
                    )
                    val_dataloader = torch.utils.data.DataLoader(
                        val_dataloader.dataset,
                        batch_size=val_dataloader.batch_size,
                        shuffle=self.shuffle,
                        num_workers=val_dataloader.num_workers,
                        pin_memory=True,
                    )
                    logger.info(
                        "Sequence length increased to: {}".format(
                            current_sequence_len
                        )
                    )
                    if bumped.reset_warmups:
                        logger.info("Resetting KL & auto & noise warm-ups")
                else:
                    logger.info("Unknown early stop condition")
                # Window bumps do not advance this count. The increment is
                # after this epoch's loss, so the trigger epoch keeps the
                # previous subband and decorr scale.
                if bumped.kind in ("auto", "kl"):
                    auto_kl_done += 1
                patience_after = warmup.active_patience(
                    auto_kl_done,
                    current_auto_warm,
                    current_noise_warm,
                    current_kl_warm,
                    current_sequence_len,
                )
                changes = []
                if bumped.kind == "window":
                    changes.append(("window", old_sequence, float(current_sequence_len)))
                    # auto_first also resets KL and the autonomous fraction.
                    # Those resets are their own rows so each vertical line
                    # is in the table.
                    if float(current_kl_warm) != old_kl:
                        changes.append(("KL", old_kl, float(current_kl_warm)))
                    if float(current_auto_warm) != old_auto:
                        changes.append(("auto", old_auto, float(current_auto_warm)))
                    if float(current_noise_warm) != old_noise:
                        changes.append(("noise", old_noise, float(current_noise_warm)))
                elif bumped.kind == "kl":
                    changes.append(("KL", old_kl, float(current_kl_warm)))
                elif bumped.kind == "auto":
                    if bumped.auto_bumped:
                        changes.append(("auto", old_auto, float(current_auto_warm)))
                    if bumped.noise_bumped:
                        changes.append(("noise", old_noise, float(current_noise_warm)))
                # The scale used next epoch. A window bump does not move
                # auto_kl_done, so these rows stay empty then. The row's
                # epoch is the one that decided the bump; the figure draws
                # the line on the following epoch, where the series changes.
                for kind, enabled, warmup_on, old_scale in (
                    (
                        "subband",
                        self._regularizers.subband_on,
                        self._regularizers.subband_warmup,
                        sub_scale,
                    ),
                    (
                        "decorr",
                        self._regularizers.decorr_on,
                        self._regularizers.decorr_warmup,
                        dec_scale,
                    ),
                ):
                    if not enabled or not warmup_on:
                        continue
                    new_scale = warmup_progress(
                        auto_kl_done,
                        warmup.autonomous_steps,
                        warmup.kl_steps,
                        True,
                    )
                    if abs(new_scale - float(old_scale)) > 1e-12:
                        changes.append((kind, float(old_scale), new_scale))
                for kind, old_value, new_value in changes:
                    row = make_bump_row(
                        epoch,
                        kind,
                        old_value,
                        new_value,
                        patience_before,
                        patience_after,
                        cpt_at_trigger,
                    )
                    bump_events.append(row)
                    logger.info(format_bump_line(row))
            if self.tie_noise_to_auto:
                current_noise_warm = current_auto_warm
            cpt_patience_epochs[epoch] = cpt_patience
            best_state_epochs[epoch] = cur_best_epoch

            # Save model parameters regularly
            if epoch % save_frequency == 0:
                # Full prefix through this epoch, not only up to the best
                # epoch, so the file matches the figure just drawn and a
                # reload can restore every series. The first write is
                # before the figures, so a crash while plotting still
                # leaves the weights. The second write, after the plot
                # timer, puts this epoch's time_plot into the checkpoint.
                history_path = os.path.join(save_dir, "loss_model.pckl")
                save_file = os.path.join(save_dir, self.model_name + "_checkpoint.pt")

                def write_epoch_checkpoint():
                    loss_log = history_at(epoch + 1)
                    save_training_history(history_path, loss_log)
                    torch.save(
                        {
                            "epoch": cur_best_epoch,
                            "best_val_loss": best_val_loss,
                            "cpt_patience": cpt_patience,
                            "model_state_dict": best_state_dict,
                            "optim_state_dict": best_optim_dict,
                            "loss_log": loss_log,
                        },
                        save_file,
                    )

                write_epoch_checkpoint()
                logger.info(
                    "Epoch: {} ===> checkpoint stored with current best epoch: {}".format(
                        epoch, cur_best_epoch
                    )
                )

                # Save the loss figure
                sync_cuda(self.device)
                plot_t0 = time.perf_counter()
                save_figures_dir = os.path.join(save_dir, "vis_during_training")
                if not os.path.exists(save_figures_dir):
                    os.makedirs(save_figures_dir)

                # Markers from the saved prefix. The unfilled tail is not a change.
                shown = epoch + 1
                kl_warm_epochs = warmup_change_epochs(kl_warm_values[:shown])
                auto_warm_epochs = warmup_change_epochs(auto_warm_values[:shown])
                noise_warm_epochs = warmup_change_epochs(noise_warm_values[:shown])
                sequence_len_epochs = warmup_change_epochs(sequence_len_values[:shown])
                # Disabled terms pass None, so the warm-up panel and the
                # bump lines leave them out.
                subband_line_epochs = (
                    warmup_change_epochs(subband_warm_values[:shown])
                    if self._regularizers.subband_on
                    else None
                )
                decorr_line_epochs = (
                    warmup_change_epochs(decorr_warm_values[:shown])
                    if self._regularizers.decorr_on
                    else None
                )

                visualize_total_loss(
                    train_loss[: epoch + 1],
                    val_loss[: epoch + 1],
                    kl_warm_epochs,
                    auto_warm_epochs,
                    noise_warm_epochs,
                    sequence_len_epochs,
                    self.model_name,
                    self.autonomous_sampling_method,
                    self.noise_sampling_method,
                    save_figures_dir,
                    tag,
                    subband_warm_epochs=subband_line_epochs,
                    decorr_warm_epochs=decorr_line_epochs,
                )
                visualize_val_criteria(
                    self._val_criteria,
                    val_loss[: epoch + 1],
                    val_spectrum[: epoch + 1],
                    val_criteria_score[: epoch + 1],
                    kl_warm_epochs,
                    auto_warm_epochs,
                    noise_warm_epochs,
                    sequence_len_epochs,
                    self.model_name,
                    self.autonomous_sampling_method,
                    self.noise_sampling_method,
                    save_figures_dir,
                    tag,
                    best_epoch=(
                        cur_best_epoch
                        if 0 <= cur_best_epoch <= epoch
                        else None
                    ),
                    include_spectrum=self._monitor_spectrum,
                    drift=(
                        {
                            name: drift_arrays["val_" + name][: epoch + 1]
                            for name in DRIFT_PLOT_NAMES + VARNORM_DRIFT_METRICS
                        }
                        if self._monitor_spectrum
                        else None
                    ),
                    recon_k=(
                        drift_arrays["val_recon_k"][: epoch + 1]
                        if self._monitor_spectrum
                        else None
                    ),
                    lyapunov=(
                        {
                            "ftle": drift_arrays["val_ftle"][: epoch + 1],
                            "lle_mean": drift_arrays["val_lle_mean"][: epoch + 1],
                            "lle_std": drift_arrays["val_lle_std"][: epoch + 1],
                            "lle_gt": self._lle_gt,
                        }
                        if self._monitor_spectrum
                        else None
                    ),
                    subband_warm_epochs=subband_line_epochs,
                    decorr_warm_epochs=decorr_line_epochs,
                )
                # Drawn on every save_frequency checkpoint, whenever the
                # fixed windows were scored, including when drift is not
                # a selection criterion. The arrays are not checkpointed.
                if self._latest_drift_steps is not None:
                    drift_figures = save_drift_timestep_figures(
                        self._latest_drift_steps,
                        save_figures_dir,
                        epoch,
                        flip=self._spectrum_tf_steps,
                        varnorm_terms=self._latest_drift_steps_varnorm,
                    )
                    logger.info(
                        "Epoch: {} ===> drift timestep figures {}".format(
                            epoch, ", ".join(drift_figures)
                        )
                    )
                visualize_total_loss(
                    train_loss[: epoch + 1],
                    val_loss[: epoch + 1],
                    kl_warm_epochs,
                    auto_warm_epochs,
                    noise_warm_epochs,
                    sequence_len_epochs,
                    self.model_name,
                    self.autonomous_sampling_method,
                    self.noise_sampling_method,
                    save_figures_dir,
                    tag,
                    log_scale=True,
                    subband_warm_epochs=subband_line_epochs,
                    decorr_warm_epochs=decorr_line_epochs,
                )

                if self.model_name in ["VRNN", "MT_VRNN"]:
                    visualize_recon_loss(
                        train_recon[: epoch + 1],
                        val_recon[: epoch + 1],
                        kl_warm_epochs,
                        auto_warm_epochs,
                        noise_warm_epochs,
                        sequence_len_epochs,
                        self.model_name,
                        self.autonomous_sampling_method,
                        self.noise_sampling_method,
                        save_figures_dir,
                        tag,
                        subband_warm_epochs=subband_line_epochs,
                        decorr_warm_epochs=decorr_line_epochs,
                    )
                    visualize_recon_loss(
                        train_recon[: epoch + 1],
                        val_recon[: epoch + 1],
                        kl_warm_epochs,
                        auto_warm_epochs,
                        noise_warm_epochs,
                        sequence_len_epochs,
                        self.model_name,
                        self.autonomous_sampling_method,
                        self.noise_sampling_method,
                        save_figures_dir,
                        tag,
                        log_scale=True,
                        subband_warm_epochs=subband_line_epochs,
                        decorr_warm_epochs=decorr_line_epochs,
                    )
                    visualize_kld_loss(
                        train_kl[: epoch + 1],
                        val_kl[: epoch + 1],
                        kl_warm_epochs,
                        auto_warm_epochs,
                        noise_warm_epochs,
                        sequence_len_epochs,
                        self.model_name,
                        self.autonomous_sampling_method,
                        self.noise_sampling_method,
                        save_figures_dir,
                        tag,
                        subband_warm_epochs=subband_line_epochs,
                        decorr_warm_epochs=decorr_line_epochs,
                    )
                    visualize_kld_loss(
                        train_kl[: epoch + 1],
                        val_kl[: epoch + 1],
                        kl_warm_epochs,
                        auto_warm_epochs,
                        noise_warm_epochs,
                        sequence_len_epochs,
                        self.model_name,
                        self.autonomous_sampling_method,
                        self.noise_sampling_method,
                        save_figures_dir,
                        tag,
                        log_scale=True,
                        subband_warm_epochs=subband_line_epochs,
                        decorr_warm_epochs=decorr_line_epochs,
                    )

                visualize_combined_metrics(
                    delta_per_epoch[: epoch + 1],
                    delta_threshold,
                    kl_warm_epochs,
                    kl_warm_values[: epoch + 1],
                    auto_warm_epochs,
                    auto_warm_values[: epoch + 1],
                    noise_warm_epochs,
                    noise_warm_values[: epoch + 1],
                    sequence_len_epochs,
                    sequence_len_values[: epoch + 1],
                    cpt_patience_epochs[: epoch + 1],
                    best_state_epochs[: epoch + 1],
                    self.model_name,
                    self.autonomous_sampling_method,
                    self.noise_sampling_method,
                    save_figures_dir,
                    tag,
                    patience_threshold_epochs[: epoch + 1],
                    windows_kept=windows_kept[: epoch + 1],
                    windows_dropped_nan=windows_dropped_nan[: epoch + 1],
                    windows_dropped_shift=windows_dropped_shift[: epoch + 1],
                    gap_splits=gap_split_counts[: epoch + 1],
                    mean_start_shift=mean_start_shift[: epoch + 1],
                    dataset_stats=dataset_stats,
                    timing={
                        "time_train": time_train[: epoch + 1],
                        "time_val": time_val[: epoch + 1],
                        "time_val_forward": time_val_forward[: epoch + 1],
                        "time_spectrum": time_spectrum[: epoch + 1],
                        "time_drift": time_drift[: epoch + 1],
                        "time_rosenstein": time_rosenstein[: epoch + 1],
                        "time_plot": time_plot[: epoch + 1],
                        "time_eval": time_eval[: epoch + 1],
                    },
                    subband_warm_values=(
                        subband_warm_values[:shown]
                        if self._regularizers.subband_on
                        else None
                    ),
                    subband_loss_weight_values=(
                        subband_loss_weight_values[:shown]
                        if self._regularizers.subband_on
                        else None
                    ),
                    decorr_warm_values=(
                        decorr_warm_values[:shown]
                        if self._regularizers.decorr_on
                        else None
                    ),
                    decorr_loss_weight_values=(
                        decorr_loss_weight_values[:shown]
                        if self._regularizers.decorr_on
                        else None
                    ),
                    subband_warm_epochs=subband_line_epochs,
                    decorr_warm_epochs=decorr_line_epochs,
                )

                if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
                    visualize_sigma_history(
                        sigmas_history[:, : epoch + 1],
                        self.model_name,
                        save_figures_dir,
                        tag,
                        kl_warm_epochs,
                        (
                            auto_warm_epochs
                            if self.autonomous_sampling_method in ["ss", "sm"]
                            else None
                        ),
                        (
                            noise_warm_epochs
                            if self.noise_sampling_method in ["ss", "sm"]
                            else None
                        ),
                        sequence_len_epochs,
                        subband_warm_epochs=subband_line_epochs,
                        decorr_warm_epochs=decorr_line_epochs,
                    )
                    visualize_alpha_history(
                        sigmas_history[:, : epoch + 1],
                        self.model_name,
                        save_figures_dir,
                        tag,
                        kl_warm_epochs,
                        (
                            auto_warm_epochs
                            if self.autonomous_sampling_method in ["ss", "sm"]
                            else None
                        ),
                        (
                            noise_warm_epochs
                            if self.noise_sampling_method in ["ss", "sm"]
                            else None
                        ),
                        sequence_len_epochs,
                        subband_warm_epochs=subband_line_epochs,
                        decorr_warm_epochs=decorr_line_epochs,
                    )

                # Same sigma/alpha history figures for PLRNN / shPLRNN diagonal A.
                save_plrnn_timescale_figures(
                    None
                    if a_sigmas_history is None
                    else a_sigmas_history[:, : epoch + 1],
                    self.model_name,
                    save_figures_dir,
                    tag,
                    kl_warm_epochs,
                    (
                        auto_warm_epochs
                        if self.autonomous_sampling_method in ["ss", "sm"]
                        else None
                    ),
                    (
                        noise_warm_epochs
                        if self.noise_sampling_method in ["ss", "sm"]
                        else None
                    ),
                    sequence_len_epochs,
                    visualize_sigma_history=visualize_sigma_history,
                    visualize_alpha_history=visualize_alpha_history,
                    subband_warm_epochs=subband_line_epochs,
                    decorr_warm_epochs=decorr_line_epochs,
                )

                # Visualize model parameters
                visualize_combined_parameters(
                    name_values=current_named_params,
                    explain="epoch_{}".format(epoch),
                    save_path=os.path.join(
                        save_figures_dir, "params_and_grads", "model_parameters_matrix"
                    ),
                )
                visualize_combined_parameters(
                    name_values=current_named_params,
                    explain="epoch_{}".format(epoch),
                    save_path=os.path.join(
                        save_figures_dir, "params_and_grads", "model_parameters_line"
                    ),
                    matrix_or_line="line",
                )
                # Visualize gradients
                visualize_combined_parameters(
                    name_values=current_named_params_grad,
                    explain="epoch_{}_gradients".format(epoch),
                    save_path=os.path.join(
                        save_figures_dir, "params_and_grads", "model_gradients_matrix"
                    ),
                    showing_gradient=True,
                    gradient_clip_value=self.gradient_clip,
                )
                visualize_combined_parameters(
                    name_values=current_named_params_grad,
                    explain="epoch_{}_gradients".format(epoch),
                    save_path=os.path.join(
                        save_figures_dir, "params_and_grads", "model_gradients_line"
                    ),
                    matrix_or_line="line",
                    showing_gradient=True,
                    gradient_clip_value=self.gradient_clip,
                )
                # Visualize clipped gradients
                if clipped_named_params_grad is not None:
                    visualize_combined_parameters(
                        name_values=clipped_named_params_grad,
                        explain="epoch_{}_clipped_gradients".format(epoch),
                        save_path=os.path.join(
                            save_figures_dir, "params_and_grads", "model_gradients_line"
                        ),
                        matrix_or_line="line",
                        showing_gradient=True,
                        gradient_clip_value=self.gradient_clip,
                    )
                # Longer training figure. Cap to the longest in-subject
                # span so an empty window set does not crash next(iter).
                _viz_len = cap_eval_seq_len(train_dataloader.dataset, 1000)
                _viz_ready = False
                if _viz_len < 1:
                    logger.info(
                        "Skipping long-sequence training visualization: "
                        "no subject can supply a window"
                    )
                else:
                    try:
                        train_dataloader.dataset.update_sequence_length(_viz_len)
                        _viz_ready = len(train_dataloader.dataset.data_idx) > 0
                        if not _viz_ready:
                            logger.info(
                                "Skipping long-sequence training visualization: "
                                "no window of length {}".format(_viz_len)
                            )
                    except ValueError as _viz_exc:
                        logger.info(
                            "Skipping long-sequence training visualization: {}".format(
                                _viz_exc
                            )
                        )
                        _viz_ready = False
                if not _viz_ready and _viz_len >= 1:
                    train_dataloader.dataset.update_sequence_length(
                        current_sequence_len
                    )
                if _viz_ready:
                    first_batch = next(iter(train_dataloader))
                    temp_batch_data = first_batch.permute(1, 0, 2).to(self.device)

                    # Extract missing mask for this batch
                    temp_missing_mask = None
                    observation_process = getattr(
                        dataset_config, "observation_process", None
                    )

                    if (
                        self._is_indicate_observation(observation_process)
                        and temp_batch_data.size(2) >= 2
                    ):
                        # Interleaved [x,m,...]: odd dims are is_observed → per-signal missing
                        is_observed = temp_batch_data[:, :, 1::2]
                        temp_missing_mask = (is_observed < 0.5).float()

                    elif hasattr(train_dataloader.dataset, "missing_mask"):
                        # Fallback for other observation processes
                        seq_len_temp, batch_size_temp, x_dim_temp = temp_batch_data.shape
                        batch_start_idx = (
                            train_dataloader.dataset.data_idx[0]
                            if len(train_dataloader.dataset.data_idx) > 0
                            else 0
                        )
                        batch_end_idx = min(
                            batch_start_idx + seq_len_temp,
                            len(train_dataloader.dataset.missing_mask),
                        )
                        missing_mask_slice = train_dataloader.dataset.missing_mask[
                            batch_start_idx:batch_end_idx
                        ]

                        # Convert to tensor and expand to batch size
                        if isinstance(missing_mask_slice, np.ndarray):
                            temp_missing_mask = (
                                torch.from_numpy(missing_mask_slice)
                                .float()
                                .unsqueeze(1)
                                .expand(-1, batch_size_temp, -1)
                            )
                        else:
                            temp_missing_mask = missing_mask_slice.unsqueeze(1).expand(
                                -1, batch_size_temp, -1
                            )

                    # Reset sequence length back to current_sequence_len
                    train_dataloader.dataset.update_sequence_length(current_sequence_len)

                    # SART: Compute autonomous mode selector for visualization #################
                    model_mode_selector = create_autonomous_mode_selector(
                        temp_batch_data.size(0),
                        mode=autonomous_config["mode"],
                        autonomous_ratio=autonomous_config["autonomous_ratio"](),
                        batch_size=temp_batch_data.size(1),
                        x_dim=dataset_config.x_dim,
                        device=self.device,
                    )
                    # replace where nan is present with 1 to fill with autonomous mode
                    model_mode_selector = torch.where(
                        temp_batch_data.isnan().detach(),
                        torch.tensor(1.0, device=self.device),
                        model_mode_selector,
                    )
                    # END: Compute autonomous mode selector for visualization #################

                    # START: Compute noise-mixing #####################
                    noise_config = noise_configs.get(self.noise_sampling_method)
                    noise_selector = None
                    if noise_config is not None:
                        noise_selector = create_autonomous_mode_selector(
                            temp_batch_data.size(0),
                            mode=noise_config["mode"],
                            autonomous_ratio=noise_config["ratio_fn"](),
                            batch_size=temp_batch_data.size(1),
                            x_dim=dataset_config.x_dim,
                            device=self.device,
                        )
                        if self.noise_target == "tf_only":
                            noise_selector *= 1 - model_mode_selector
                        elif self.noise_target in ["ar_only", "hybrid_ar"]:
                            noise_selector *= model_mode_selector
                    # END: Compute noise-mixing #####################

                    # Visualize teacher forcing and autonomous mode
                    visualize_teacherforcing_2_autonomous(
                        temp_batch_data,
                        self.model,
                        auto_mode_selector=model_mode_selector,
                        noise_selector=noise_selector,
                        save_path=os.path.join(
                            save_figures_dir, "sequence_reconstruction", "training_mode"
                        ),
                        explain=f"training_epoch:{epoch}_klwarm{current_kl_warm}_auto_warm{current_auto_warm}_window{current_sequence_len}",
                        inference_mode=True,
                        missing_mask=temp_missing_mask,
                        hide_mask_output=(
                            self._is_indicate_observation(
                                getattr(dataset_config, "observation_process", None)
                            )
                        ),
                    )

                    # START: Compute autonomous mode selector for Even Bursts visualization #################
                    model_mode_selector_flip_test = create_autonomous_mode_selector(
                        temp_batch_data.size(0),
                        mode="even_bursts",
                        autonomous_ratio=0.1,
                        device=self.device,
                        batch_size=temp_batch_data.size(1),
                        x_dim=dataset_config.x_dim,
                    )
                    model_mode_selector = torch.where(
                        temp_batch_data.isnan().detach(),
                        torch.tensor(1.0, device=self.device),
                        model_mode_selector,
                    )
                    # END: Compute autonomous mode selector for Even Bursts visualization #################
                    # START: Compute noise-mixing #####################
                    noise_config = noise_configs.get(self.noise_sampling_method)
                    noise_selector = None
                    if noise_config is not None:
                        noise_selector = create_autonomous_mode_selector(
                            temp_batch_data.size(0),
                            mode=noise_config["mode"],
                            autonomous_ratio=noise_config["ratio_fn"](),
                            batch_size=temp_batch_data.size(1),
                            x_dim=dataset_config.x_dim,
                            device=self.device,
                        )
                        if self.noise_target == "tf_only":
                            noise_selector *= 1 - model_mode_selector
                        elif self.noise_target in ["ar_only", "hybrid_ar"]:
                            noise_selector *= model_mode_selector
                    # END: Compute noise-mixing #####################

                    visualize_teacherforcing_2_autonomous(
                        temp_batch_data,
                        self.model,
                        auto_mode_selector=model_mode_selector_flip_test,
                        noise_selector=noise_selector,
                        save_path=os.path.join(
                            save_figures_dir, "sequence_reconstruction", "even_bursts"
                        ),
                        explain=f"even_bursts_epoch:{epoch}_klwarm{current_kl_warm}_auto_warm{current_auto_warm}_window{current_sequence_len}",
                        inference_mode=True,
                        missing_mask=temp_missing_mask,
                        hide_mask_output=(
                            self._is_indicate_observation(
                                getattr(dataset_config, "observation_process", None)
                            )
                        ),
                    )

                    # START: Compute autonomous mode selector for Half-Half visualization #################
                    model_mode_selector_flip_test = create_autonomous_mode_selector(
                        temp_batch_data.size(0),
                        mode="half_half",
                        device=self.device,
                        batch_size=temp_batch_data.size(1),
                        x_dim=dataset_config.x_dim,
                    )
                    model_mode_selector = torch.where(
                        temp_batch_data.isnan().detach(),
                        torch.tensor(1.0, device=self.device),
                        model_mode_selector,
                    )
                    # END: Compute autonomous mode selector for Half-Half visualization #################
                    # START: Compute noise-mixing #####################
                    noise_config = noise_configs.get(self.noise_sampling_method)
                    noise_selector = None
                    if noise_config is not None:
                        noise_selector = create_autonomous_mode_selector(
                            temp_batch_data.size(0),
                            mode=noise_config["mode"],
                            autonomous_ratio=noise_config["ratio_fn"](),
                            batch_size=temp_batch_data.size(1),
                            x_dim=dataset_config.x_dim,
                            device=self.device,
                        )
                        if self.noise_target == "tf_only":
                            noise_selector *= 1 - model_mode_selector
                        elif self.noise_target in ["ar_only", "hybrid_ar"]:
                            noise_selector *= model_mode_selector
                    # END: Compute noise-mixing #####################

                    visualize_teacherforcing_2_autonomous(
                        temp_batch_data,
                        self.model,
                        auto_mode_selector=model_mode_selector_flip_test,
                        noise_selector=noise_selector,
                        save_path=os.path.join(
                            save_figures_dir, "sequence_reconstruction", "half_half"
                        ),
                        explain=f"half_half_epoch:{epoch}_klwarm{current_kl_warm}_auto_warm{current_auto_warm}_window{current_sequence_len}",
                        inference_mode=True,
                        missing_mask=temp_missing_mask,
                        hide_mask_output=(
                            self._is_indicate_observation(
                                getattr(dataset_config, "observation_process", None)
                            )
                        ),
                    )

                if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
                    alphas = sigmoid_10(sigmas_history[:, epoch])
                    logger.info(
                        "alphas: {}".format([f"{alpha:.5f}" for alpha in alphas])
                    )
                log_plrnn_timescales(
                    logger,
                    a_sigmas_history,
                    epoch,
                    alongside_mt=self.model_name in ("MT_RNN", "MT_VRNN"),
                )
                sync_cuda(self.device)
                time_plot[epoch] = time.perf_counter() - plot_t0
                write_epoch_checkpoint()
                logger.info(
                    "Epoch: {} ===> checkpoint loss_log updated with plot wall time {:.4f}".format(
                        epoch, time_plot[epoch]
                    )
                )

            # Current weights, in addition to the best checkpoint and the
            # final best-epoch file. 0 / off leaves this path unused.
            if self._checkpoint_every and (epoch + 1) % self._checkpoint_every == 0:
                epoch_file = os.path.join(
                    save_dir, "{}_epoch{}.pt".format(self.model_name, epoch)
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "val_criteria": ",".join(self._val_criteria),
                        "model_state_dict": self.model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    },
                    epoch_file,
                )
                logger.info(
                    "Epoch: {} ===> current-epoch checkpoint stored at {}".format(
                        epoch, epoch_file
                    )
                )

            log_epoch_wall(epoch)

        if ran_epoch and not stopped_early:
            logger.info(format_epoch_budget_line(epoch))

        if ran_epoch and not self._any_selection_eligible:
            # No epoch ever had a finite spectrum, so nothing was eligible
            # for selection. Keep the last weights instead.
            message = (
                "NO EPOCH HAD A FINITE SPECTRUM SCORE: model selection never "
                "ran. Falling back to the last checkpoint (epoch {}) as the "
                "final model.".format(epoch)
            )
            logger.error(message)
            print("[Learning Algo] WARNING: " + message)
            warnings.warn(message, RuntimeWarning, stacklevel=1)
            best_state_dict = self.model.state_dict()
            cur_best_epoch = epoch

        # Save the final weights of network with the best validation loss
        save_file = os.path.join(
            save_dir, self.model_name + "_final_epoch" + str(cur_best_epoch) + ".pt"
        )
        torch.save(best_state_dict, save_file)

        # Save the training loss and validation loss, including every series
        # the training figures use. Slice after the copy so the final alpha
        # log still reads the last trained epoch.
        loss_file = os.path.join(save_dir, "loss_model.pckl")
        save_training_history(loss_file, history_at(epoch + 1))
        logger.info("Loss saved in: {}".format(loss_file))
        train_loss = train_loss[: epoch + 1]
        val_loss = val_loss[: epoch + 1]
        train_recon = train_recon[: epoch + 1]
        train_kl = train_kl[: epoch + 1]
        val_recon = val_recon[: epoch + 1]
        val_kl = val_kl[: epoch + 1]
        if self.model_name == "MT_RNN" or self.model_name == "MT_VRNN":
            sigmas_history = sigmas_history[:, : epoch + 1]
        if a_sigmas_history is not None:
            a_sigmas_history = a_sigmas_history[:, : epoch + 1]

        # Log all final alpha values
        if self.model_name in ["MT_RNN", "MT_VRNN"]:
            alphas = sigmoid_10(sigmas_history[:, -1])
            logger.info("Final alphas: {}".format([f"{alpha:.5f}" for alpha in alphas]))
        log_plrnn_timescales(
            logger,
            a_sigmas_history,
            -1,
            final=True,
            alongside_mt=self.model_name in ("MT_RNN", "MT_VRNN"),
        )

        # run evaluation script
        eval_file = os.path.join(project_root, "src", "dvae", "eval", "eval_signal.py")
        sync_cuda(self.device)
        eval_t0 = time.perf_counter()
        subprocess.run([sys.executable, eval_file, "--saved_dict", save_file])
        sync_cuda(self.device)
        if ran_epoch:
            time_eval[epoch] = time.perf_counter() - eval_t0
            logger.info(
                "Eval inside training wall_s={:.4f}".format(time_eval[epoch])
            )
            save_training_history(loss_file, history_at(epoch + 1))


# sigmoid function for arrays
def sigmoid_10(x):
    x = np.array(x)
    return 1 / (1 + 10 ** (-x))


# reverse of sigmoid function for arrays


def sigmoid_reverse_10(y):
    y = np.array(y)
    return -np.log10((1 / y) - 1)


def get_norm_from_named_params(named_params):
    norm = 0
    for _, param in named_params:
        norm += np.linalg.norm(param)
    return norm
