# Professor Forcing baseline (Lorenz63)

Opt-in trainer for [Professor Forcing](https://arxiv.org/abs/1610.09038) and a labeled hidden-state matching ablation. It does not go through `LearningAlgorithm`, `bin/train_model.py`, or `bin/generate_config_file_for_multiple.py`.

What matches the paper, and every Lorenz-specific choice, is in [FORMULATION.md](FORMULATION.md).

## Train one config

```sh
python bin/train_professor_forcing.py --cfg config/lorenz63/cfg_professor_forcing.ini
```

`variant = professor_forcing` is the paper. `variant = hidden_match` is the ablation: same teacher-forced and free-running passes, with `||h_free - stopgrad(h_TF)||^2` in place of the discriminator. `hidden_match` is not part of Lamb et al.

`free_run_feedback = mean` feeds the readout back in (the default). `gaussian` adds `observation_std` noise during training only. Evaluation rollouts always feed the mean.

## Comparison grid and Deigo

```sh
python bin/generate_professor_forcing_configs.py
bash scripts/slurm/baselines/run_professor_forcing.sh professor_forcing_lorenz63
```

The generator writes gitignored ini files under `config/lorenz63/generated_professor_forcing/`. The active grid is fully observed and `Markov_AvgLen15_0.5` masks, observation processes `only_x`, `all_xyz`, and `only_x_indicate`, both section-4.1 behavior sequences (`pre_tanh`, `pre_tanh_and_output`), both generator objectives (`nll_cf`, `nll_cf_ct`), and the `hidden_match` ablation. Edit the lists at the top of `bin/generate_professor_forcing_configs.py` to add `only_xy`, `only_xz`, or the other Markov rates.

The Slurm script is the CPU `compute` partition and the same Singularity paths as `scripts/slurm/train/run_training_multiple.sh`. Each job runs `bin/train_professor_forcing.py`. Checkpoints and `metrics.json` go under the device `saved_root` in `professor_forcing/<experiment>/`.

## Metrics

Validation records `mse_tf`, `mse_auto`, `spectrum_error_tf`, `spectrum_error_auto`, `kld_tf`, and `kld_auto`, plus the `_mean` aliases used by the aggregate tables. They call:

- `calc_masked_mse`
- `frequency_analysis._spectrum_pair_errors` (smoothed Hellinger spectrum error)
- `compute_delay_embedding` and `state_space_kl(..., use_gmm=True)` with Lorenz `tau, m` from `_get_delay_params`
- the `half_half` flip from `create_autonomous_mode_selector_1d`

`training_metrics.png` uses `aggregate_plot_style` (display names, y-axis scale, save helper).
