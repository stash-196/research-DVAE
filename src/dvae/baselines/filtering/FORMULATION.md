# Filtering imputation baselines

Comparison methods for opportunistic teacher forcing (OTF) on masked time
series. Nothing here is registered in the training loop.

## What is scored

`eval_signal` reports, on the free-run timesteps of a half-and-half schedule:

- `mse_tf` / `mse_auto` from `calc_masked_mse` (missing positions excluded)
- `spectrum_error_tf` / `spectrum_error_auto` (Hellinger distance of smoothed spectra)
- `kld_tf` / `kld_auto` (delay-embedding Gaussian-mixture KL)

This package calls those same helpers (`metrics_adapter.py`). The teacher-forced
slot is the one-step predictive mean while observations are still assimilated
where they exist. The autonomous slot stops assimilating at the flip and rolls
the transition forward. That matches an OTF net, which emits a prediction and
then consumes the observation only when the mode selector says so and the
sample is present.

`mse_impute_filtered` and `mse_impute_smoothed` are extra. They are the error
on missing positions only. The shared MSE helper does not compute that.

Not wired, on purpose: free-run EMD and Rosenstein Lyapunov (open PR adding
`free_run_metrics.py`), and Jacobian / Lyapunov metrics that need a torch
module inside `eval_signal.py`. Window-0 traces are saved as
`window0_traces.npz` so those hooks can be pointed at the arrays later without
editing the eval loop.

## 1. Linear-Gaussian Kalman EM

Model (Shumway & Stoffer, 1982):

```
x_0 ~ N(mu0, P0)
x_t = F x_{t-1} + w_t,   w_t ~ N(0, Q)
y_t = H x_t + v_t,       v_t ~ N(0, R)
```

A time with no observation keeps the predicted state (Kalman gain zero), which
is the missing-data modification in that paper. If only some components are
present, the update uses those rows of `H` and `R`. That partial-vector case
is the textbook extension; the 1982 paper treats a time index as entirely
present or entirely missing.

The RTS smoother and the lag-one covariance `Cov(x_t, x_{t-1} | y) = P_{t|T} J_{t-1}^T`
feed the EM M-step. `H` is optional. With `H` fixed at `I`, `F`, `Q`, and `R`
are identifiable. If `H` is free there is a change-of-basis ambiguity, so
parameter tables should not be compared entrywise to a ground-truth `F`.

`R` is diagonal by default. Each observed component then has its own
closed-form regression M-step. A full `R` is updated only on times where every
component is present, because a non-diagonal `R` couples missing components
into the complete-data likelihood and the 1982 paper does not give that
M-step. The filter still assimilates partial times either way.

Per-component standardization is a numerical rescaling. Under `H = I` it
replaces `F` by `D^{-1} F D` when the component scales differ.

Outputs:

- `predicted`: `H x_{t|t-1}` (OTF teacher-forced analogue)
- `filtered`: `H x_{t|t}` (causal imputation; equals the prediction when the
  whole vector is missing)
- `smoothed`: `H x_{t|T}` (RTS, acausal)
- `autonomous`: `H x_{t|t-1}` with assimilation turned off after the flip

pykalman and statsmodels are not used. pykalman's EM path does not take an
arbitrary per-component mask in a way that matches this M-step, and a required
dependency would change existing environments. The implementation is NumPy.

## 2. Lorenz oracle EKF / UKF

`role = oracle`. Not a fair competitor.

The discrete map is the sequential forward Euler in
`dvae.dataset.generate_save_lorenz_63.L63.step`: `x` is updated before it is
used in the `y` and `z` increments. Parameters come from the dataset folder
name (`sigma10_rho28_beta8d3_..._dt0.01`) or from the CLI. The generator
source currently sets `rho = 40` while the dataset default folder is `rho = 28`;
the label on the files being scored wins.

`Q` and `R` are small fixed variances. The simulated system is deterministic;
they only keep the covariance from collapsing. They are not learned.

The backward pass is an extended RTS smoother (EKF) or the unscented RTS step
that uses the sigma-point cross-covariance (UKF). Tresp & Hofmann discuss the
EKF as a filter; they do not write this backward pass. It is only used for the
oracle.

## 3. Tresp & Hofmann nonlinear baseline

Source that was actually read: Tresp & Hofmann, "Nonlinear Time-Series
Prediction with Missing and Noisy Data", Neural Computation 10, 731–747, 1998
(`https://www.dbs.ifi.lmu.de/~tresp/papers/nc_pap_p2.pdf`). The 1995 IEEE
Neural Networks for Signal Processing paper is the conference version. Its
training section was not available as a PDF here. The 1998 paper says training
under missing data "will be addressed in a companion paper (Tresp and Hofmann,
1997)" and lists that companion as a manuscript in preparation. A 2003 author
note says they never followed it up. Those training equations are not
implemented and were not guessed.

What is implemented from the 1998 paper:

- Equation (1) and the appendix: `y_t = f(y_{t-1}, ..., y_{t-N}) + eps` with
  Gaussian `eps`, and the first-order multivariate form `y_t = f(y_{t-1}) + eps`.
  Missing entries are the infinite measurement-noise case of section 3, done
  by skipping those coordinates.
- Section 4.1: Gibbs sampling. The univariate conditional is not Gaussian
  because `f` is nonlinear, so candidates are drawn by
  sampling-importance-resampling. The proposal is the parent conditional
  `N(f(parents), sigma^2)`. The weight is the product of successor likelihoods.
- Section 4.2: maximum-likelihood substitution. Missing values minimize the
  sum of squared one-step residuals (equation 9, joint over the missing set).
  Long gaps are optimized in chunks of 400 scalars. The paper states one joint
  optimization and does not discuss chunking.
- Section 4.3: extended Kalman filter, first-order linearization, online.
  A present observation uses a tiny measurement variance instead of exact zero,
  because `R = 0` makes the innovation covariance singular. The paper uses zero
  for a perfect measurement. Lag copies in a delay state are deterministic in
  the paper; a `1e-8` jitter is added on those coordinates so `P` stays
  positive definite.
- Section 5.1: K-step prediction by forward simulation. `autonomous_sampled`
  averages `f` over noisy rollouts. `autonomous` is the noiseless iteration,
  which is the OTF free-run and the "iterated system" the paper plots against
  the Monte Carlo mean. They are different once `f` is nonlinear and the
  horizon is longer than one step.

Univariate series use delay `N` (default 3, a Lorenz-style embedding, not a
number from the paper). Multivariate series use delay 1, the appendix's
first-order state. A delay stack of a multivariate series is not written out
in the paper and is rejected.

Training default `observed_targets` is the sunspot protocol in section 5.1.2:
least squares on windows whose inputs and target are fully observed, plus a
weight-decay penalty, with `sigma^2` set to the average squared residual. The
paper does not name the optimizer; this code uses Adam with classical
`weight_decay`. It does not name the hidden-unit nonlinearity; the net is one
tanh hidden layer, as in the Bishop 1994 reference they cite. `hidden=0` is a
linear map for tests. It is not a configuration from the paper. Their sunspot
weight decay of 0.2 is that experiment's setting, not the default here
(default 0).

`train_mode=multiple_imputation` is off by default. It alternates the
observed-target fit with the section 4.1 Gibbs fill. The paper calls averaging
several imputations "multiple imputation" (Little & Rubin) in section 4.1.
This loop is that idea plus the conditionals they write. It is not a
transcription of the unpublished 1997 manuscript.

The first `delay` samples have no parent window. The paper leaves
`P(y_1, ..., y_N)` unspecified, so those positions stay at the interpolated
fill and are not sampled.

A bootstrap particle filter is available as `particle` (`n_particles > 0`).
It is not in the 1998 equations. The author's 2003 comment says particle
filters replaced the Gibbs sampler they had proposed. It is that later
substitute, not the primary trajectory.

## Deviations checklist

| Item | Status |
| --- | --- |
| Shumway & Stoffer filter, RTS, EM | Implemented |
| Partial-vector observations | Textbook extension; 1982 paper is all-or-nothing per time |
| Diagonal `R` when components go missing independently | Closed-form M-step; full `R` ignores partial times in the M-step only |
| Tresp 1998 prediction (Gibbs/SIR, ML substitution, EKF, K-step sampling) | Implemented |
| Tresp 1995/1997 training equations | Not available; not invented |
| Training that the 1998 experiments report | `observed_targets` |
| Multiple-imputation training loop | Optional, documented as a reconstruction |
| Lorenz EKF/UKF | Oracle only |
