# Training

This document explains how training is implemented in the codebase and how the main training loop behaves.

**Entry point**
- **Script**: The training run is started from train_model.py. It:
  - Parses CLI args via `Options().get_params()` (config_utils.py).
  - Loads device paths and assigns a `job_id`.
  - Instantiates `LearningAlgorithm(params=params)` and calls `learning_algo.train()`.

**Configuration & CLI**
- **CLI args**: `--cfg` (required), `--use_pretrain`, `--pretrain_dict`, `--reload`, `--model_dir`, `--job_id`. See `Options.get_params()` in config_utils.py.
- **Config file**: A standard INI-style config parsed by `myconf()` (imported in `dvae.utils`) controls dataset, network, and training hyperparameters (sequence lengths, sampling method, warm-up limits, lr, optimization, etc.). The config path is passed via `--cfg`.

**High-level flow**
1. Load config and parameters (dataset name, model name, sampling method, device, batch size, sequence length, etc.).
2. Create/save an experiment directory (unless resuming).
3. Build the model via `LearningAlgorithm.build_model()` (calls model builders such as `build_VRNN`, `build_RNN`, `build_MT_RNN`, `build_MT_VRNN`).
4. Initialize optimizer via `LearningAlgorithm.init_optimizer()` — supports Adam/AdamW; for multi-task models it separates parameter groups (e.g., `sigmas` with a different lr).
5. Build data loaders with `build_dataloader()` from dataset_builder.py, which returns train and validation DataLoaders for the chosen dataset.
6. Enter epoch loop and run mini-batch training + validation with warm-up schedules, early stopping, checkpointing and visualizations.

**Data loading**
- **Dataset config**: `DatasetConfig` (dataclass) holds `data_dir`, `x_dim`, `batch_size`, `seq_len`, `observation_process`, `with_nan`, etc. The unified loader builder is in dataset_builder.py.
- **Shapes**: Datasets produce tensors of shape `(batch_size, seq_len, x_dim)`; the training code permutes to `(seq_len, batch_size, x_dim)` before passing to the model.

**Mode selectors: teacher forcing / autonomous / noise**
- The training supports mixed teaching/autonomous (scheduled sampling, mixed sampling, even bursts, etc.). The per-time-step mode selector is created with `create_autonomous_mode_selector(...)` (from `dvae.utils.model_mode_selector`).
- A separate `noise_selector` can be generated to inject stochastic noise according to a noise-sampling strategy.
- If NaNs exist in the input batch (missing data), the code overlays autonomous mode at those positions to prevent teacher forcing on missing inputs.

**Per-batch forward/backward**
- Forward:
  - Model is called as `recon_batch_data = model(batch_data, mode_selector=..., noise_selector=...)`.
  - The reconstruction predictions are compared to ground-truth using masked losses (see next section).
- Loss components:
  - Reconstruction: MSE (`loss_MSE`) with flexible masking modes:
    - `none`: compute loss for any observed values.
    - `strict`: compute only where the base mode selector used pure TF.
    - `weighted`: weight reconstruction loss by teacher-forcing ratio.
  - KL: `loss_KLD` for stochastic models (VRNN, MT_VRNN); set to zero for pure RNNs.
  - Total loss per batch: average reconstruction + (beta * KL * kl_warm_coeff).
- Backward & optimization:
  - `loss_tot_avg.backward()` then gradient clipping (if configured) via `torch.nn.utils.clip_grad_norm_`.
  - `optimizer.step()` updates parameters.
  - The training logs parameter/gradient norms each step for debugging/monitoring.

**Warm-ups, annealing and scheduling**
- The code supports several warm-up and annealing mechanisms:
  - **Autonomous warm-up (scheduled-sampling style)**: controlled by `autonomous_sampling_method`, `auto_warm_start`, `auto_warm_limit` (anneals `current_auto_warm` from start to limit).
  - **Noise warm-up**: separate schedule for additive/noise sampling (can be tied to autonomous warm-up).
  - **KL warm-up**: for VRNN-like models the KL weight is annealed from 0 → 1 (controlled by `kl_warm` variables and `beta`).
  - **Sequence-length warm-up**: training can start with a shorter `initial_sequence_len` and bump to the configured `sequence_len` progressively.
- Warm-up adjustments occur when validation improvement stalls: the code uses an early-stop patience counter; when patience triggers, if warm-ups are not complete it increases warm-ups (autonomous / noise / KL) or increases sequence length, instead of stopping immediately. This allows training to continue while progressively exposing the model to harder training regimes.

**Validation, early stopping & checkpointing**
- A validation pass runs after the training pass to compute validation reconstruction/KL and total loss (same loss calculations as training).
- The script tracks best validation loss and uses `delta_threshold` to determine improvements.
- **Early stopping**: If validation does not improve for `early_stop_patience` epochs and all warm-ups/sequence length are complete, training stops.
- **Checkpointing**:
  - Regular checkpoints are saved every `save_frequency` epochs; saved items include best model state, optimizer state, epoch, and loss logs.
  - Final best model weights are saved as `<ModelName>_final_epoch<best_epoch>.pt`.
  - A `loss_model.pckl` file with training/validation loss traces and warm-up epochs is saved at the end.

**Visualization & diagnostics**
- At checkpoint time the routine saves multiple visual artifacts (loss curves, KL curves, reconstruction curves, parameter/gradient visualizations, teacher-forcing vs autonomous mode visualizations) using the visualizers in `dvae.visualizers`.
- The training logs parameter norms and gradient norms each step; clipped gradients are logged and visualized when clipping is active.
- For multi-task models (MT_RNN/MT_VRNN) the code tracks `sigmas_history` and derives `alphas` for logging/plots.

**Missing-data handling**
- If inputs contain NaNs, `model_mode_selector` is forced to autonomous for those entries (so the model is not forced to use missing ground-truth).
- For datasets with `observation_process == "only_x_indicate"`, a special masking baseline is supported: mask channels are forced to teacher forcing while losses are computed only on the signal channel.

**Resuming training**
- If `--reload` is passed and `--model_dir` is provided or inferred from the config file folder, the code loads a checkpoint (`<model_name>_checkpoint.pt`), restores model & optimizer states, and resumes from `start_epoch` while padding loss arrays to the epoch target.

**Final evaluation**
- After saving the final best model, the training routine launches an evaluation script (eval_signal.py) with the saved model file to run the final evaluation pipeline.



# Preprocessing XHRO
(Preprocessing XHRO is done in another repository than this one) 
**Overview**

- **Purpose:** Prepare raw XHRO multimodal vital signals for RNN training by cleaning, resampling to a uniform grid, masking/artifact removal, filtering, segmenting, optionally computing coarse features, and saving cleaned time series. The implementation lives in xhro_mne_dataset.py.
- **Inputs:** CSV raw recordings with `datetime` and signal columns (e.g., `ch1..ch4`, `mag`, `phase`, `temp`, validity flags).
- **Outputs:** Per-recording cleaned time-series parquet (`filtered_data.parquet`), diagnostic plots and logs, and optional coarse-feature parquet.

**Key files**
- Main pipeline: xhro_mne_dataset.py
- Low-level processing helpers: utils.py
- Visual diagnostic helpers: visualizers.py

**High-level steps (what the script does)**

- **Load & initial cleaning**
  - Uses `load_data()` to read CSV and optionally save raw parquet.
  - Runs `describe_data()` and `replace_invalid_with_nan()` to canonicalize invalid values to NaN.
  - Removes obviously identical-channel artifacts with `replace_identical_channels_vals_with_nan()`.

- **Pre-visualization & logging**
  - Creates multiple time-series plots and NaN diagnostics via visualizers.
  - Writes a preprocessing log to `preprocessing.log`.

- **Uniform resampling (critical for RNNs)**
  - Target sampling freq: `sfreq = 250` Hz → `dt = 1 / sfreq` (0.004s).
  - Builds a uniform datetime grid from `t_min` to `t_max` with step `dt`.
  - Uses `pd.merge_asof(..., direction='nearest', tolerance=tol)` to align original samples to the uniform grid, where `tol` defaults to 0.002s (2 ms). Loss per channel is logged and large loss warns.
  - Asserts the merged timeline has near-uniform spacing (mean and std within tolerance).

- **Masking & NaN-ing artifacts**
  - Masks EEG/ECG regions contaminated by bio-impedance (`mask_eeg_ecg_during_bio_impedance()`).
  - Replaces flat or zero-spread regions with NaN (`replace_zero_spread_with_nan()`).
  - Replaces burst artifacts with NaN (`replace_burst_artifacts_with_nan()`).
  - Applies global outlier removal (`replace_global_outlier_relative_threshold_with_nan()`).
  - Recomputes first-valid index and drops pre-leading-NaN region.

- **Per-segment notch & band-pass filtering**
  - Detects continuous valid segments with `find_continuous_segments()` and uses MNE `RawArray` for filtering.
  - Applies per-segment notch filters (`apply_notch_to_segments()`).
  - For ECG channels: band-pass roughly 0.05–40 Hz.
  - For EEG channels: band-pass roughly 0.5–80 Hz.
  - Filtered values replace DataFrame channel columns.

- **Fine-grained bad-subsection detection**
  - Runs `replace_bad_subsections_in_segments_with_nan()` to find short subsections with relative amplitude outliers, kurtosis anomalies, or excessive per-window outlier fraction and marks them NaN.

- **Post-processing visuals and summaries**
  - Re-creates diagnostic plots (time-series, NaN heatmap, segment-length histograms).
  - Logs NaN statistics before/after preprocessing.

- **Feature computation (optional, commented)**
  - Code includes commented examples to compute coarse features (`compute_features()` / `compute_band_powers()`), with a suggested window `n = 5` seconds and possible features: band powers, total power, relative band power, peak frequency, spectral entropy, Hjorth parameters.
  - The coarse feature pivot and parquet saving are shown in comments for downstream experiments.

- **Save**
  - Final cleaned/resampled DataFrame saved to `filtered_data.parquet` under the recording output directory.

**Important parameters & defaults (what to report in thesis)**

- **Channels & types:** `channels = ["ch1","ch2","ch3","ch4"]`; `ecg_channels = ["ch1","ch2"]`; `eeg_channels = ["ch3","ch4"]`.
- **Sampling:** `sfreq = 250 Hz` (target), `dt = 0.004 s`; merge tolerance `tol = 0.002 s`.
- **Filtering:** Notch applied per segment; ECG band-pass `0.05–40 Hz`; EEG band-pass `0.5–80 Hz`.
- **Segment min lengths:** ECG min ~5 s (int(5*sfreq)), EEG min ~10 s (int(10*sfreq)).
- **Artifact thresholds:** Examples in code: `relative_amp_thresh=10.0`, `mad_multiplier=7.0`, `abs_thresh=100_000` — these are tunable hyperparameters used to NaN-out outliers.
- **Zero-spread detection:** `steps_before=20` / `steps_after=20` (samples) default in calls — used to detect and NaN flat lines.

**How outputs map to training for RNNs (practical guidance)**

- **Sequence sampling rate:** Use the pipeline output sampling rate `250 Hz` (or downsample later). For many RNN tasks, downsample to e.g. 50–125 Hz to reduce sequence length while preserving relevant frequency bands.
- **Windowing for RNN inputs:**
  - Fixed-length sliding windows: e.g., 5 s windows → 5 * 250 = 1250 timesteps.
  - Stride: choose 50–100% overlap (commonly 50%) for more training samples.
  - For shorter continuous segments, only sample windows fully contained within valid (non-NaN) segments or allow masked timesteps (see below).
- **Handling NaNs / missing timesteps:**
  - Preferred: discard windows with >X% NaN (e.g., >10%). This is already accommodated by the optional `nan_threshold` in `compute_features()`.
  - Alternative: impute per-channel (linear interpolation, forward/backward fill, or model-based), or keep NaN and provide a mask input channel to the RNN.
  - The pipeline offers `fill_zero_spread_with_linear_interpolation()` and other helpers for interpolation strategies.
- **Normalization:**
  - Use per-channel robust normalization (e.g., median and MAD) — utilities include `robust_normalize()`.
  - Normalize on training set statistics and apply same transform to validation/test.
- **Feature engineering options:**
  - Raw multichannel time series: shape (batch, timesteps, channels) for sequence models.
  - Coarse features: compute band powers, spectral entropy, Hjorth parameters per window and feed as lower-rate features to the RNN (or to an auxiliary input).
  - Combined approach: feed raw sensor-channel sequences to the RNN and coarse features to a parallel dense branch.
- **Labels & supervised targets:** Not included in preprocessing; ensure labels align temporally to window centers or windows. For event-based labels, compute label presence per-window.

**Data shapes & batching (example)**

- Raw windows: `X` shape → `(batch_size, T, C)` where `T = window_seconds * sfreq` (e.g., `5s * 250 = 1250`), `C = 4` (ch1–ch4) or include derived channels/features.
- Mask input (optional): `(batch_size, T, C)` boolean mask where 1 = valid, 0 = NaN; feed to mask-aware RNNs or multiply/time-gate inputs.
- Coarse features: `(batch_size, T_coarse, F)` with `T_coarse = window_seconds / coarse_window` (if using stacked temporal features).

**Recommendations for RNN training (thesis-ready advice)**

- Use a reproducible pipeline: store `filtered_data.parquet`, logs, and plots per recording as produced by the script; version-control the preprocessing code and parameter set.
- Data augmentation: add small Gaussian noise, random channel dropout, or random time-shifts to improve robustness.
- Batch sampling: sample windows uniformly from experiments to avoid subject/session imbalance.
- Validation split: split by recording/subject to avoid leakage (i.e., don’t split within the same contiguous recording).
- Regularization: dropout, weight decay, and early stopping are useful given physiological variability.
- Learning rate and sequence length: longer sequences increase GPU memory; experiment with downsampling or using multi-scale models (raw at higher rate + condensed features).
- Mask-aware loss: if you allow windows with masked timesteps, use masked loss computations (ignore NaN entries).

**Reproducibility & logging**

- Logs are saved to `preprocessing.log` in each recording output directory.
- Diagnostic figures (time series, NaN heatmaps, and segment-length histograms) are saved to plots and interactive HTMLs to `htmls/`.
- Keep the exact parameter values (sfreq, tol, thresholds) in an experiment config file or in the thesis appendix.

**Notes about limitations & tuning**

- The current merge uses nearest-neighbor within `tol`. If many samples are lost, consider iterative fallback strategies: increase `tolerance`, perform linear interpolation between measured timestamps, or resample with careful jitter handling.
- Artifact detection thresholds are heuristic; report sensitivity analyses in the thesis (e.g., how varying `mad_multiplier` or `relative_amp_thresh` affects retained data).
- If using per-segment MNE filtering, ensure edge effects are considered (filter transients at segment boundaries).

**Short example: turning pipeline outputs into RNN training data**

1. Run preprocessing to produce `filtered_data.parquet`.
2. For each recording:
   - Load cleaned dataframe, select `channels = [ch1,ch2,ch3,ch4]`.
   - Extract contiguous non-NaN segments (minimum length L).
   - Generate sliding windows of length `T=window_seconds*sfreq` with stride `s`.
   - Discard windows with >10% NaN (or impute and add mask).
   - Normalize windows using training-set medians/MADs.
   - Batch into `(batch, T, C)` tensors and feed to RNN training loop.



# DataLoading

**Data loading: Lorenz63 & XHRO**

- **Location (classes):** `Lorenz63` is implemented in `src/dvae/dataset/lorenz63_dataset.py` and `Xhro` in `src/dvae/dataset/xhro_dataset.py`.

- **Common contract:** both dataset classes expose the same Dataset-like API used by `build_dataloader()`:
  - `__len__()` returns the number of available sequence start indices.
  - `__getitem__(i)` returns a contiguous slice `seq[start : start+seq_len]` as a numpy/torch array with shape `(seq_len, x_dim)`.
  - `update_sequence_length(new_seq_len, ...)` recomputes the list of valid `data_idx` start indices for the current split (train/valid/test).
  - `get_missing_mask(index)` and `get_full_xyz(index)` provide access to the original full signal and the mask of NaNs for diagnostics.

- **Lorenz63 specifics:**
  - Loads precomputed arrays from pickled files (train/test variants) containing complete trajectories. Optionally a synthetic mask (e.g., `mask_*_pnan...`) can be loaded and applied so missing values are represented as `NaN` in `the_sequence`.
  - `missing_mask` is extracted before any observation processing so the dataset preserves where values were truly missing in the raw data.
  - `apply_observation_process(...)` supports options like `xyz_to_xyz`, `only_x`, `only_x_interpolate`, `only_x_indicate` and simple noise injections. The latter two produce 1D or 2D outputs (e.g., `[x_imputed, is_observed]`).
  - After observation processing the implementation infers `x_dim` if unset, converts to sliding windows if `overlap=True`, and sets `data_idx` via `update_sequence_length()`.

- **XHRO specifics:**
  - Reads cleaned recordings from parquet files in `.../xhro/processed/<dataset_label>/` (either high-frequency `original` columns or `coarsed` feature tables). The class also sets a `sampling_freq` (e.g., 250 Hz) when reading.
  - Uses `select_observation_window_slice()` to crop long recordings to experimental windows when a `dataset_label` maps to a time slice.
  - Extracts `missing_mask` from the raw sequence before applying the selected `observation_process` so the original NaN pattern is preserved.
  - `apply_observation_process(...)` handles multiple modes: selecting/coarsening columns, linear interpolation (`only_x_interpolate`), and `only_x_indicate` which returns a two-channel array `[x_imputed, is_observed]` where the second channel is an explicit observation indicator used by masking baselines.
  - For 1D inputs the class supports overlapping moving-window creation via `create_moving_window_sequences()` using numpy strides for efficiency.
  - `update_sequence_length()` accepts an optional `minimum_nan_ratio` when temporarily increasing window size for visualization or filtering; indices that exceed the NaN ratio threshold are discarded.

- **Shapes & integration with training loop:**
  - Datasets yield samples shaped `(seq_len, x_dim)`; `DataLoader` batches these into `(batch_size, seq_len, x_dim)`.
  - The training code expects `(seq_len, batch_size, x_dim)` and therefore permutes batches before model call. This permutation is important to preserve the time-first RNN input contract.
  - `missing_mask` from the datasets is used by the training loop to: (1) overlay autonomous mode where inputs are NaN (prevent TF on missing values), and (2) compute masked reconstruction losses (the dataset `observation_process == 'only_x_indicate'` is handled specially by `_prepare_masked_loss`).

- **Practical differences to note (for thesis):**
  - Lorenz63 is simulated / offline data stored as numpy pickles with optional synthetic masks; XHRO is real, timestamped, parquet-serialized physiological data with richer observation-processing (interpolation, coarsening, explicit is_observed channel).
  - XHRO stores and exposes `sampling_freq` and maintains time-window slicing utilities (useful when reasoning about continuous segment lengths and filtering operations in preprocessing).

The above details explain how the two dataset classes prepare sequences, preserve missing-data information, and expose the Dataset API consumed by the training loop.


# Evaluation Metrics

The post-training evaluation is orchestrated by `src/dvae/eval/eval_signal.py`. It computes robust indicators of generative quality and dynamically visualizes both signal tracking and structural fidelity. Multiple mathematically grounded metrics evaluate model performance between teacher-forced (TF) and autonomous (Auto) conditions.

### 1. Mean Squared Error (MSE)
A standard pointwise reconstruction metric applied under both teacher-forcing and autonomous prediction tasks. It evaluates how closely the predicted trajectory tracks the ground truth at each time step.

$$ MSE = \frac{1}{T \times D} \sum_{t=1}^{T} \sum_{d=1}^{D} (\hat{y}_{t,d} - y_{t,d})^2 $$
where $T$ is the sequence length, $D$ is the dimensionality of the signal, $y$ is the ground truth, and $\hat{y}$ is the network's prediction. The evaluation captures $MSE_{TF}$ (error when fully teacher-forced) and $MSE_{Auto}$ (error when running fully autonomously without correction).

### 2. Power Spectrum Error
While pointwise metrics like MSE diverge rapidly for chaotic systems, spectral topology reveals structural fidelity in the frequency domain. The power spectrum error measures the discrepancy between the frequency components of the ground truth and the autonomously predicted signal.

Let $\mathcal{P}_{y}(f)$ and $\mathcal{P}_{\hat{y}}(f)$ denote the power spectral densities of the ground truth and the predicted signal respectively. The metric computes the normalized pointwise error across the frequency spectrum. 

This captures whether the model learns the correct underlying frequencies (i.e. oscillations and periodicities) even if the exact phase drifts over time.

### 3. State Space Divergence (KLD)
Measures the structural consistency of the learned embedding state-space across differing sampling strategies. The script specifically evaluates the Kullback-Leibler (KL) Divergence (or an approximated discrete formulation) between the state distribution experienced under teacher forcing versus pure autonomous generation.

Lower KLD indicates that the autonomous trajectory remains well-supported mathematically within the manifold of the teacher-forced training distribution, rather than diverging into unsampled feature spaces.

### 4. Local Drift Statistics ($\Delta MSE$)
This is a specialized metric built to validate implicit regularization and drift stability empirically. It measures how quickly the autonomous trajectory drifts away from the teacher-forced trajectory precisely after the teacher signal is dropped (a "fork point").

Let:
- $y_{true}$ = the ground truth target.
- $y_{TF}$ = the prediction under teacher forcing.
- $e = y_{TF} - y_{true}$ = the prediction error under teacher forcing.
- $y_{Auto}$ = the autonomous prediction after the fork point.
- $d = y_{Auto} - y_{TF}$ = the drift vector induced by lack of teacher context.

The error of the autonomous prediction is $y_{Auto} - y_{true} = d + e$.
The change in Mean Squared Error upon dropping the teacher is:
$$ \Delta MSE = \|d + e\|^2 - \|e\|^2 = \|d\|^2 + 2d^T e $$

**Interpretation:**
- $\|d\|^2$ is the **drift magnitude**: how much the model deviates from its own teacher-forced path.
- $2d^T e$ is the **cross-term**: the correlation between the drift and the underlying error.
- **Self-correction**: A small drift magnitude coupled with a *negative* cross-term ($d^T e < 0$) indicates self-correction; the model naturally drifts in a direction that opposes the TF error.
- **Error Amplification**: A large drift magnitude or a positive cross-term indicates error amplification when operating autonomously.

During evaluation, the script computes these vectors over 20-40 step windows following the fork, exposing the precise transient stability behavior of the model. 

### Visualizations
During evaluation, the script produces several specialized diagnostics:
1. **Embedding Spaces:** It projects the high-dimensional hidden states ($h_t$) into 3D using methods such as Kernel PCA, ICA, and t-SNE, visually distincting TF (green) from Autonomous (red) conditions.
2. **Variable Evolution:** Plots hidden state variance across time.
3. **Sequence Reconstruction:** Visually contrasts prediction capabilities under pure teacher-forcing, even bursts, and half-half conditions.



