# TODO

Working list folded from [PhD Thesis Research Notes](https://www.notion.so/PhD-Thesis-Research-Notes-2ef8b066105380ca94cceacea67a0ac8) (open items as of 2026-08-18) and the live DVAE fork. Closed Notion checkboxes are omitted.

## Now

- [ ] **Lag-Operator SSM** (frequency-preserving basis)
  - [ ] Fix numerical instability in the recurrence
  - [ ] Test on XHRO and VitalPatch
- [ ] **XHRO packet loss**
  - [ ] Re-run preprocessing (missed artifacts)
  - [ ] Train and compare realtime vs recovered
- [ ] **OTF**
  - [ ] Train more 4-D input RNNs on XHRO
  - [ ] Multi-dimension evaluation (code change already in tree)
  - [ ] Compare against interpolate and mask-variable (indicate) baselines
  - [ ] More public medical / wearable datasets
  - [ ] Kalman filtering capability
  - [ ] Latent-variable forcing capability
- [ ] Land uncommitted XhroProper path: `xhro_proper_dataset.py`, `tests/test_xhro_proper.py`, dataset_builder / packet-loss / SLURM / config-generator edits
  - Parity vs `Xhro` / `XhroPacketLoss` (same seq_len, split, observation names)
  - Do not mix parquet and grok NPZ corpora in one tensor



## Thesis writing (`Doc:`)

From GS / DoyaS feedback and the 2026-06-03 / 06-26 must-fix list.

- [ ] RNN experiment figure caption inconsistencies
- [ ] XHRO per-channel missing rates
- [ ] Why chaos (motivation in the text)



## Experiments still open

- [ ] **TF vs window schedule** (which curriculum is actually best?)
  1. Max window, max TF (TF effect only)
  2. Min window, max TF (hybrid)
  3. Min window, no TF (window effect only)
  - Evaluate small-window training on large-window test
- [ ] ch2 at each `p_auto`; also try `ss`
- [ ] Lorenz: re-run for statistics
- [ ] Heatmaps: unify color ranges
- [ ] Benchmarks still to put in the paper: interpolation; zero-fill + masking indicator
- [ ] Parameter-count comparisons (done in code; keep in figures)
- [ ] HiPPO baseline (prediction + oscillation)
- [ ] Long-range evaluation
- [ ] Visualize / regularize d^\top e (drift–error cross term)
- [ ] Extra datasets if the story needs them: MIMIC-III / PhysioNet (GRU-D), meteorological DA vs AR, van der Schaar clinical sets
- [ ] Suntory / SIC: multi-dimensional input (DoyaS); new datasets (Mizutani); LagOpSSM + Dynamix for analysis automation / foundation model



## Theory

- [ ] Treat TF vs autonomous as the claim, not an implementation detail
  - VRNN closed-loop still uses q(z \mid \hat y, h), not prior p(z \mid h) — decide if that is intended
- [ ] Regularizer: uncorrelated d and e; or (d^\top e)^2; or drop the cross term and train \ell^d + \ell^{\mathrm{TF}} directly
- [ ] Can ergodic expectation be replaced by noise-forcing expectation?
- [ ] Interpret learned \alpha against physical timescales (Lorenz `true_alphas`, SHO periods, damped \gamma, XHRO bands)
- [ ] Prefer Durstewitz metrics (Hellinger spectra, delay-embed KL, \DeltaMSE drift) over Auto MSE on chaos



## Code

- [ ] `LearningAlgorithm.build_model` only wires RNN / VRNN / MT_RNN / MT_VRNN — restore DKF/SRNN/… or archive them off the live path
- [ ] Dataset registry: document SHO / DampedSHO; sinusoid is referenced in train but not in `DATASET_REGISTRY`
- [ ] Document `loss_mask_mode` (`none` / `strict` / `weighted`) as an experimental factor
- [ ] Noise mixer: keep or drop (current sweeps often `noise_sampling_method=none`)
- [ ] Deigo loop: generate configs → `run_training_multiple.sh` → copy `/flash` → `/bucket` → eval/aggregate
- [ ] Paths in `config/device_paths.yaml` (Studio cache ≠ live cluster mount)



## Paper-shaped next experiments

- [ ] Lorenz `only_x` + `ptf` sweep: does MT-\alpha recover the attractor under partial obs?
- [ ] Same architecture on XHRO packet loss vs recovered (retrans)
- [ ] Indicate vs interpolate vs model-impute (no future leak)