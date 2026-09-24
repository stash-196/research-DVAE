"""Post-training Jacobian summaries and Benettin Lyapunov spectrum.

Matches the discrete-map diagnostics Hess et al. (ICML 2023) run on the
shallow PLRNN, without pulling Julia or DynamicalSystems.jl into this repo.

Autonomous map
--------------
Lyapunov exponents here are for an autonomous system. Hess et al. reject
driven trajectories: during the QR average the external input does not
depend on time. The map is

    z_{t+1} = R(z_t, u),   u = feature_extractor(0)

``R`` is ``model.recurrence`` (the cell, plus the MT-RNN α mix when the
model is ``MT_RNN``). The observation fed to the model is the zero vector.
Decoder outputs stay out of the next input, so this is the latent map and
not the observation-closed loop. A non-zero external drive is the case
their Lyapunov path rejects.

When ``dense_x`` is empty the feature extractor is the identity, so the
recurrent cell's external input is exactly 0 and ``linear_C`` contributes
only its bias (Durstewitz's ``h1``). A non-trivial feature extractor still
sees x = 0; ``u = feature(0)`` is a constant offset. It moves the orbit
but does not add a ∂u/∂z term, so the analytic Jacobian below is unchanged.

Teacher-forced warmup only chooses the initial condition z0 (to land near
the data). Those steps are not accumulated. ``n_transient`` further
autonomous steps (DynamicalSystems' ``Ttr``) are evolved, including the
tangent QR, and then discarded before the average.

Shallow PLRNN Jacobian (Hess / Durstewitz)
------------------------------------------
For z_{t+1} = A⊙z + W1 φ(W2 z + h2) + h1 with φ = ReLU and h2 = ``b2``::

    J(z) = diag(A) + W1 @ diag(W2 z + b2 > 0) @ W2

which is the same gate as ``diag(W2 z > -h2)``. The clipped shallow basis
φ(W2 z + h2) - φ(W2 z) uses the gate ``(W2 z > -h2) - (W2 z > 0)``. This
repo's ``shPLRNN.forward`` is the plain ReLU, not the clipped basis, so
eval always uses ``clipped=False``.

Dendritic PLRNN (``PLRNN``)::

    J(z) = diag(A) + W_off @ diag(z > 0)

with the diagonal of W removed, matching ``PLRNN.forward``.

MT-RNN α mix, applied on top of the cell map f::

    J = diag(1 - α) + diag(α) @ J_f

Vanilla tanh RNN cells use the analytic tanh Jacobian. Anything else that
is still a single deterministic recurrence falls back to
``torch.autograd.functional.jacobian``. LSTM, VRNN, and other unsupported
architectures are skipped with a warning so eval still writes its summary.

Benettin spectrum
-----------------
Not the training-time ``opnorm(J)`` used to set a GTF α (that path is
intentionally absent). This is the post-training estimator: wrap the
autonomous map the way DynamicalSystems.jl wraps a
``DeterministicIteratedMap`` + ``TangentDynamicalSystem`` with a hand
Jacobian, then ``lyapunovspectrum(ds, N)`` (Benettin 1980 / Geist QR):

    Y = J(z) @ Q;  Q, R = qr(Y);  λ += log|diag(R)|

averaged over ``n_steps`` after the transient. Local TF-vs-Auto drift
(``local_drift_analysis.py``) is a separate statistic and is left as-is.
"""

from __future__ import annotations

import inspect
import math
import warnings
from typing import Any, Mapping, MutableMapping, Optional

import torch
import torch.nn.functional as F


DEFAULT_N_STEPS = 1000
DEFAULT_N_TRANSIENT = 100
DEFAULT_N_WARMUP = 50

AUTONOMOUS_INPUT = "external_observation_zero"

_ANALYTIC_CELLS = ("shPLRNN", "PLRNN", "RNN")
_TINY = torch.finfo(torch.float64).tiny


class UnsupportedLatentMap(RuntimeError):
    """The model has no single autonomous latent map this eval can differentiate."""


def _row_batch(h: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if h.ndim == 1:
        return h.unsqueeze(0), True
    if h.ndim == 2:
        return h, False
    raise ValueError(
        f"state must have shape (M,) or (B, M), got {tuple(h.shape)}"
    )


def jacobian_shplrnn(
    h: torch.Tensor,
    A: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    clipped: bool = False,
) -> torch.Tensor:
    """Analytic Jacobian of the shallow PLRNN latent map.

    Parameters match the in-repo cell: ``A`` is (M,), ``W1`` is (M, L),
    ``W2`` is (L, M), ``b2`` is (L,) (Durstewitz ``h2``). ``h`` is (M,) or
    (B, M). Returns (M, M) or (B, M, M).

    ``clipped=False`` is φ = ReLU(W2 z + b2), gate ``W2 z + b2 > 0``.
    ``clipped=True`` is the Durstewitz clipped basis
    ReLU(W2 z + b2) - ReLU(W2 z), gate ``(W2 z > -b2) - (W2 z > 0)``.
    ReLU's kink uses the same convention as ``torch.relu`` (derivative 0
    at 0), i.e. a strict ``>``.
    """
    rows, squeeze = _row_batch(h)
    A = A.reshape(-1)
    b2 = b2.reshape(-1)
    hidden = A.numel()
    latent_sh = b2.numel()
    if W1.shape != (hidden, latent_sh) or W2.shape != (latent_sh, hidden):
        raise ValueError(
            f"shPLRNN shapes A {tuple(A.shape)}, W1 {tuple(W1.shape)}, "
            f"W2 {tuple(W2.shape)}, b2 {tuple(b2.shape)} are inconsistent"
        )
    if rows.shape[-1] != hidden:
        raise ValueError(
            f"state dim {rows.shape[-1]} does not match A ({hidden})"
        )

    pre = rows @ W2.T + b2
    gate = (pre > 0).to(dtype=rows.dtype)
    if clipped:
        # (W2 z > -b2) - (W2 z > 0), with b2 playing the role of h2.
        gate = gate - (rows @ W2.T > 0).to(dtype=rows.dtype)
    # W1 @ diag(gate) @ W2, batched over rows.
    weighted = W1.unsqueeze(0) * gate.unsqueeze(1)
    J = torch.diag(A).to(dtype=rows.dtype, device=rows.device).unsqueeze(0)
    J = J + weighted @ W2
    return J[0] if squeeze else J


def jacobian_plrnn(
    h: torch.Tensor,
    A: torch.Tensor,
    W: torch.Tensor,
    W_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Analytic Jacobian of the dendritic PLRNN.

    ``J = diag(A) + W_off @ diag(h > 0)``. ``W_mask`` zeros the diagonal,
    as in ``PLRNN.forward``. Without a mask, ``W`` is used as-is.
    """
    rows, squeeze = _row_batch(h)
    A = A.reshape(-1)
    hidden = A.numel()
    if W.shape != (hidden, hidden):
        raise ValueError(
            f"PLRNN W shape {tuple(W.shape)} does not match A ({hidden},)"
        )
    if rows.shape[-1] != hidden:
        raise ValueError(
            f"state dim {rows.shape[-1]} does not match A ({hidden})"
        )
    W_off = W if W_mask is None else W * W_mask
    gate = (rows > 0).to(dtype=rows.dtype)
    weighted = W_off.unsqueeze(0) * gate.unsqueeze(1)
    eye = torch.diag(A).to(dtype=rows.dtype, device=rows.device)
    J = eye.unsqueeze(0) + weighted
    return J[0] if squeeze else J


def jacobian_alpha_mix(J_f: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Jacobian of h -> (1-α)⊙h + α⊙f(h).

    ``J = diag(1-α) + diag(α) @ J_f`` (row-scale ``J_f`` by α).
    ``J_f`` is (M, M) or (B, M, M); ``alpha`` is (M,).
    """
    squeeze = J_f.ndim == 2
    if J_f.ndim not in (2, 3):
        raise ValueError(
            f"J_f must have shape (M, M) or (B, M, M), got {tuple(J_f.shape)}"
        )
    if squeeze:
        J_f = J_f.unsqueeze(0)
    hidden = J_f.shape[-1]
    alpha = alpha.reshape(-1).to(device=J_f.device, dtype=J_f.dtype)
    if alpha.numel() != hidden or J_f.shape[-2] != hidden:
        raise ValueError(
            f"alpha ({alpha.numel()}) does not match J_f {tuple(J_f.shape)}"
        )
    eye = torch.diag(1 - alpha).unsqueeze(0)
    J = eye + alpha.view(1, -1, 1) * J_f
    return J[0] if squeeze else J


def jacobian_tanh_rnn(h: torch.Tensor, cell: torch.nn.Module) -> torch.Tensor:
    """Analytic Jacobian of ``nn.RNN`` (tanh) at external input 0.

    ``h -> tanh(W_hh h + b_hh + b_ih)``, so
    ``J = diag(1 - tanh(pre)^2) @ W_hh``.
    """
    if getattr(cell, "nonlinearity", "tanh") != "tanh":
        raise UnsupportedLatentMap(
            f"nn.RNN nonlinearity {getattr(cell, 'nonlinearity', None)!r} "
            "has no analytic Jacobian; use autodiff"
        )
    rows, squeeze = _row_batch(h)
    W_hh = cell.weight_hh_l0.detach()
    b = torch.zeros(W_hh.shape[0], device=W_hh.device, dtype=W_hh.dtype)
    if getattr(cell, "bias_ih_l0", None) is not None:
        b = b + cell.bias_ih_l0.detach()
    if getattr(cell, "bias_hh_l0", None) is not None:
        b = b + cell.bias_hh_l0.detach()
    W_hh = W_hh.to(dtype=rows.dtype, device=rows.device)
    b = b.to(dtype=rows.dtype, device=rows.device)
    pre = rows @ W_hh.T + b
    sech2 = 1 - torch.tanh(pre) ** 2
    J = sech2.unsqueeze(-1) * W_hh.unsqueeze(0)
    return J[0] if squeeze else J


def jacobian_autodiff(step_fn, z: torch.Tensor) -> torch.Tensor:
    """Full Jacobian of ``step_fn`` via ``torch.autograd.functional.jacobian``.

    Fallback for a generic deterministic cell. ``z`` is (M,). The step is
    differentiated under ``enable_grad`` so this still works if the caller
    is inside ``torch.no_grad`` (as ``eval_signal`` is for the rest of eval).
    """
    z = z.detach().reshape(-1)

    def _f(v: torch.Tensor) -> torch.Tensor:
        return step_fn(v).reshape(-1)

    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(
            _f, z, create_graph=False, vectorize=False
        )
    return J.detach()


def _free_run_diagnostics(
    jac_fn,
    step_fn,
    z0: torch.Tensor,
    n_steps: int,
    n_transient: int = 0,
    k: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """One autonomous orbit: Benettin QR spectrum and Jacobian summaries.

    Jacobian summaries are taken on the same post-transient steps that
    enter the Lyapunov average. Transient steps still advance ``z`` and
    re-orthonormalize ``Q`` so the tangent space can align.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if n_transient < 0:
        raise ValueError(f"n_transient must be >= 0, got {n_transient}")

    z = z0.detach().reshape(-1)
    hidden = int(z.numel())
    if k is None:
        k = hidden
    if not (1 <= int(k) <= hidden):
        raise ValueError(f"k must be in [1, {hidden}], got {k}")
    k = int(k)

    Q = torch.eye(hidden, k, dtype=torch.float64, device=z.device)
    acc = torch.zeros(k, dtype=torch.float64, device=z.device)
    opnorms = torch.empty(n_steps, dtype=torch.float64, device=z.device)
    rhos = torch.empty(n_steps, dtype=torch.float64, device=z.device)

    total = n_transient + n_steps
    for t in range(total):
        J = jac_fn(z).detach()
        if J.shape != (hidden, hidden):
            raise ValueError(
                f"jac_fn returned shape {tuple(J.shape)}, expected "
                f"({hidden}, {hidden})"
            )
        if not torch.isfinite(J).all():
            raise RuntimeError(
                f"Jacobian became non-finite at autonomous step {t}"
            )
        J64 = J.to(dtype=torch.float64)
        # Benettin / Geist: Y = J(z) Q, then QR. J is at the pre-step state.
        Q, R = torch.linalg.qr(J64 @ Q, mode="reduced")
        if t >= n_transient:
            idx = t - n_transient
            diag = torch.diagonal(R).abs().clamp_min(_TINY)
            acc = acc + torch.log(diag)
            opnorms[idx] = torch.linalg.matrix_norm(J64, ord=2)
            eig = torch.linalg.eigvals(J64)
            rhos[idx] = torch.max(eig.abs())
        z = step_fn(z).detach().reshape(-1)
        if z.numel() != hidden or not torch.isfinite(z).all():
            raise RuntimeError(
                "autonomous free-run became non-finite at step "
                f"{t} (n_transient={n_transient}, n_steps={n_steps})"
            )

    spectrum, _ = torch.sort(acc / n_steps, descending=True)
    return {
        "spectrum": spectrum,
        "jac_opnorm_mean": opnorms.mean(),
        "jac_opnorm_max": opnorms.max(),
        "jac_rho_max": rhos.max(),
        "jac_rho_gt1_frac": (rhos > 1).to(dtype=torch.float64).mean(),
    }


def benettin_lyapunov_spectrum(
    jac_fn,
    step_fn,
    z0: torch.Tensor,
    n_steps: int,
    n_transient: int = 0,
    k: Optional[int] = None,
) -> torch.Tensor:
    """Lyapunov spectrum (descending) of an autonomous discrete map.

    ``jac_fn(z) -> (M, M)`` and ``step_fn(z) -> (M,)`` must be the same
    map. ``k`` is the tangent dimension (default: full spectrum). See
    the module docstring for the QR accumulation.
    """
    return _free_run_diagnostics(
        jac_fn, step_fn, z0, n_steps, n_transient=n_transient, k=k
    )["spectrum"]


def trajectory_jacobian_summaries(
    jac_fn,
    step_fn,
    z0: torch.Tensor,
    n_steps: int,
    n_transient: int = 0,
) -> dict[str, float]:
    """Mean/max ‖J‖₂, max spectral radius, and fraction of steps with ρ(J)>1.

    Computed along the autonomous free-run after ``n_transient`` steps.
    """
    diag = _free_run_diagnostics(
        jac_fn, step_fn, z0, n_steps, n_transient=n_transient, k=None
    )
    return {
        "jac_opnorm_mean": float(diag["jac_opnorm_mean"]),
        "jac_opnorm_max": float(diag["jac_opnorm_max"]),
        "jac_rho_max": float(diag["jac_rho_max"]),
        "jac_rho_gt1_frac": float(diag["jac_rho_gt1_frac"]),
    }


def _recurrence_is_deterministic(model) -> bool:
    rec = getattr(model, "recurrence", None)
    if not callable(rec):
        return False
    try:
        params = inspect.signature(rec).parameters
    except (TypeError, ValueError):
        return False
    return (
        "feature_xt" in params
        and "h_t" in params
        and "feature_zt" not in params
    )


def _uses_alpha_mix(model) -> bool:
    """True for MT_RNN's (1-α) h + α f(h) wrapper, not for a bare cell."""
    if type(model).__name__ == "MT_RNN":
        return True
    # Mixin-style override: a sigmas parameter plus alphas_per_unit, and
    # the recurrence is not BaseRNN.recurrence. VRNN is rejected earlier
    # because its recurrence takes feature_zt.
    if not hasattr(model, "sigmas") or not callable(
        getattr(model, "alphas_per_unit", None)
    ):
        return False
    rec = getattr(type(model), "recurrence", None)
    base_rec = None
    for klass in type(model).__mro__:
        if klass.__name__ == "BaseRNN" and "recurrence" in klass.__dict__:
            base_rec = klass.__dict__["recurrence"]
            break
    return rec is not None and rec is not base_rec


def _param_device_dtype(model) -> tuple[torch.device, torch.dtype]:
    try:
        param = next(model.parameters())
    except (StopIteration, AttributeError) as exc:
        raise UnsupportedLatentMap(
            "model has no parameters; Lyapunov eval skipped"
        ) from exc
    return param.device, param.dtype


def _zero_observation_feature(model, input_size: int) -> torch.Tensor:
    """Constant recurrent input corresponding to an all-zero observation."""
    device, dtype = _param_device_dtype(model)
    extractor = getattr(model, "feature_extractor_x", None)
    x_dim = getattr(model, "x_dim", None)
    if extractor is None or x_dim is None:
        return torch.zeros(input_size, device=device, dtype=dtype)
    x0 = torch.zeros(1, int(x_dim), device=device, dtype=dtype)
    with torch.no_grad():
        feat = extractor(x0)
    feat = feat.reshape(-1).detach()
    if feat.numel() != input_size:
        raise UnsupportedLatentMap(
            f"feature(x=0) has dim {feat.numel()} but the recurrent cell "
            f"expects {input_size}"
        )
    return feat


def _cell_jacobian(cell, type_rnn: str):
    """Return (jac_cell, tag) or (None, 'autodiff') for a generic cell."""
    if type_rnn == "shPLRNN":
        if not all(hasattr(cell, name) for name in ("A", "W1", "W2", "b2")):
            raise UnsupportedLatentMap(
                "type_rnn=shPLRNN but the cell is missing A/W1/W2/b2"
            )

        def jac_cell(z, cell=cell):
            return jacobian_shplrnn(
                z,
                cell.A.detach(),
                cell.W1.detach(),
                cell.W2.detach(),
                cell.b2.detach(),
                clipped=False,
            )

        return jac_cell, "analytic_shPLRNN"

    if type_rnn == "PLRNN":
        if not all(hasattr(cell, name) for name in ("A", "W")):
            raise UnsupportedLatentMap(
                "type_rnn=PLRNN but the cell is missing A/W"
            )

        def jac_cell(z, cell=cell):
            return jacobian_plrnn(
                z,
                cell.A.detach(),
                cell.W.detach(),
                getattr(cell, "W_mask", None),
            )

        return jac_cell, "analytic_PLRNN"

    if type_rnn == "RNN" and getattr(cell, "nonlinearity", "tanh") == "tanh":
        if not hasattr(cell, "weight_hh_l0"):
            return None, "autodiff"

        def jac_cell(z, cell=cell):
            return jacobian_tanh_rnn(z, cell)

        return jac_cell, "analytic_tanh_rnn"

    return None, "autodiff"


class AutonomousMap:
    """Autonomous latent step and its Jacobian, external observation = 0."""

    def __init__(self, step_fn, jac_fn, dim: int, kind: str):
        self.step_fn = step_fn
        self.jac_fn = jac_fn
        self.dim = int(dim)
        self.kind = kind

    def step(self, z: torch.Tensor) -> torch.Tensor:
        return self.step_fn(z)

    def jacobian(self, z: torch.Tensor) -> torch.Tensor:
        return self.jac_fn(z)


def build_autonomous_map(model) -> AutonomousMap:
    """Build z -> R(z, feature(0)) and a matching Jacobian.

    Raises ``UnsupportedLatentMap`` for LSTM, stochastic VRNN-style
    recurrences, and stacked ``num_rnn != 1``.
    """
    type_rnn = getattr(model, "type_rnn", None)
    num_rnn = getattr(model, "num_rnn", None)
    cell = getattr(model, "rnn", None)
    if not _recurrence_is_deterministic(model) or cell is None or type_rnn is None:
        name = type(model).__name__
        raise UnsupportedLatentMap(
            f"{name} has no deterministic single-latent recurrence "
            "(Lyapunov eval supports RNN and MT_RNN with shPLRNN, PLRNN, "
            "or tanh RNN). Skipping."
        )
    if num_rnn != 1:
        raise UnsupportedLatentMap(
            f"num_rnn={num_rnn} is not supported (only one recurrent layer). "
            "Skipping."
        )
    if type_rnn == "LSTM":
        raise UnsupportedLatentMap(
            "LSTM state is (h, c); Lyapunov eval needs a single latent "
            "state. Skipping."
        )
    if type_rnn not in _ANALYTIC_CELLS:
        raise UnsupportedLatentMap(
            f"type_rnn={type_rnn!r} has no Jacobian hook "
            f"(supported: {', '.join(_ANALYTIC_CELLS)}). Skipping."
        )

    input_size = getattr(cell, "input_size", None)
    if input_size is None:
        raise UnsupportedLatentMap(
            f"{type(cell).__name__} has no input_size; Lyapunov eval skipped"
        )
    dim = getattr(model, "dim_rnn", None)
    if dim is None:
        dim = getattr(cell, "hidden_size", None)
    if dim is None:
        raise UnsupportedLatentMap("could not read the latent dimension")

    u0 = _zero_observation_feature(model, int(input_size))
    alpha_mix = _uses_alpha_mix(model)

    def step_fn(z: torch.Tensor, model=model, u0=u0) -> torch.Tensor:
        z = z.reshape(-1)
        feat = u0.to(device=z.device, dtype=z.dtype).view(1, 1, -1)
        h = z.view(1, 1, -1)
        h_next, _ = model.recurrence(feat, h)
        return h_next.reshape(-1)

    jac_cell, jac_tag = _cell_jacobian(cell, type_rnn)
    if jac_cell is None:

        def jac_fn(z: torch.Tensor, step_fn=step_fn) -> torch.Tensor:
            return jacobian_autodiff(step_fn, z)

        jac_tag = "autodiff"
    else:

        def jac_fn(
            z: torch.Tensor,
            jac_cell=jac_cell,
            model=model,
            alpha_mix=alpha_mix,
        ) -> torch.Tensor:
            J = jac_cell(z)
            if alpha_mix:
                alpha = model.alphas_per_unit().detach().reshape(-1)
                J = jacobian_alpha_mix(J, alpha)
            return J

    kind = f"{type(model).__name__}:{type_rnn}:{jac_tag}"
    if alpha_mix:
        kind += "+alpha_mix"
    return AutonomousMap(step_fn, jac_fn, int(dim), kind)


def teacher_forced_initial_state(
    model,
    batch_data: Optional[torch.Tensor],
    n_warmup: int,
    batch_index: int = 0,
) -> tuple[torch.Tensor, int]:
    """TF warmup to pick z0. Returns ``(z0, steps_actually_run)``.

    ``batch_data`` is ``(seq, batch, x_dim)``. ``n_warmup <= 0`` or a
    missing batch starts at the origin. The warmup trajectory is not part
    of the Lyapunov average; the caller then switches to the autonomous map.
    """
    device, dtype = _param_device_dtype(model)
    dim = int(getattr(model, "dim_rnn"))
    if n_warmup <= 0 or batch_data is None:
        return torch.zeros(dim, device=device, dtype=dtype), 0
    if batch_data.ndim != 3:
        raise ValueError(
            "batch_data must have shape (seq, batch, x_dim), "
            f"got {tuple(batch_data.shape)}"
        )
    seq_len, batch_size, _ = batch_data.shape
    steps = min(int(n_warmup), int(seq_len))
    if steps <= 0:
        return torch.zeros(dim, device=device, dtype=dtype), 0
    index = int(batch_index)
    if index < 0 or index >= batch_size:
        index = 0
    x = batch_data[:steps]
    with torch.no_grad():
        mode = torch.zeros(
            steps, batch_size, x.shape[-1], device=x.device, dtype=x.dtype
        )
        model(x, mode_selector=mode, inference_mode=True, initialize_states=True)
        z0 = model.h_t[-1, index].detach().clone()
    return z0, steps


def _cfg_int(cfg, key: str) -> Optional[int]:
    if cfg is None:
        return None
    if not cfg.has_section("Evaluation") or not cfg.has_option("Evaluation", key):
        return None
    return cfg.getint("Evaluation", key)


def _resolve_int(cli_value, cfg, key: str, default: int) -> int:
    if cli_value is not None:
        return int(cli_value)
    parsed = _cfg_int(cfg, key)
    if parsed is not None:
        return int(parsed)
    return int(default)


def resolve_compute_lyapunov(cli_value, cfg=None) -> bool:
    """CLI wins when it is not None; otherwise [Evaluation] compute_lyapunov.

    Absent CLI and absent config stay off, so existing evals do not spend
    the free-run or add Jacobian keys.
    """
    if cli_value is not None:
        return bool(cli_value)
    if cfg is None:
        return False
    if not cfg.has_section("Evaluation") or not cfg.has_option(
        "Evaluation", "compute_lyapunov"
    ):
        return False
    return bool(cfg.getboolean("Evaluation", "compute_lyapunov"))


def resolve_lyapunov_settings(params: Mapping[str, Any], cfg=None) -> dict[str, Any]:
    """Resolve the eval flag and free-run lengths from CLI then config."""
    enabled = resolve_compute_lyapunov(params.get("compute_lyapunov"), cfg)
    k = _resolve_int(params.get("lyap_k"), cfg, "lyap_k", 0)
    return {
        "enabled": enabled,
        "n_steps": _resolve_int(
            params.get("lyap_n_steps"), cfg, "lyap_n_steps", DEFAULT_N_STEPS
        ),
        "n_transient": _resolve_int(
            params.get("lyap_n_transient"),
            cfg,
            "lyap_n_transient",
            DEFAULT_N_TRANSIENT,
        ),
        "n_warmup": _resolve_int(
            params.get("lyap_n_warmup"), cfg, "lyap_n_warmup", DEFAULT_N_WARMUP
        ),
        # lyap_k = 0 (the default sentinel) means the full spectrum.
        "k": None if k == 0 else int(k),
    }


def _finite_float(value: torch.Tensor, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{name} is non-finite ({number})")
    return number


def compute_lyapunov_metrics(
    model,
    batch_data: Optional[torch.Tensor] = None,
    *,
    n_steps: int = DEFAULT_N_STEPS,
    n_transient: int = DEFAULT_N_TRANSIENT,
    n_warmup: int = DEFAULT_N_WARMUP,
    k: Optional[int] = None,
    batch_index: int = 0,
) -> dict[str, Any]:
    """Run the autonomous free-run and return summary scalars.

    Raises ``UnsupportedLatentMap`` for architectures we deliberately skip.
    """
    # Validate the architecture before flipping train/eval. A second build
    # below recaches feature(x=0) once dropout is off.
    build_autonomous_map(model)
    training = bool(getattr(model, "training", False))
    model.eval()
    try:
        amap = build_autonomous_map(model)
        z0, warmup_used = teacher_forced_initial_state(
            model, batch_data, n_warmup, batch_index=batch_index
        )
        diag = _free_run_diagnostics(
            amap.jac_fn,
            amap.step_fn,
            z0,
            n_steps,
            n_transient=n_transient,
            k=k,
        )
    finally:
        if training:
            model.train()

    spectrum = [_finite_float(v, "lyap_spectrum") for v in diag["spectrum"]]
    return {
        "lyap_spectrum": spectrum,
        "lyap_max": spectrum[0],
        "jac_opnorm_mean": _finite_float(diag["jac_opnorm_mean"], "jac_opnorm_mean"),
        "jac_opnorm_max": _finite_float(diag["jac_opnorm_max"], "jac_opnorm_max"),
        "jac_rho_max": _finite_float(diag["jac_rho_max"], "jac_rho_max"),
        "jac_rho_gt1_frac": _finite_float(
            diag["jac_rho_gt1_frac"], "jac_rho_gt1_frac"
        ),
        "lyap_n_steps": int(n_steps),
        "lyap_n_transient": int(n_transient),
        "lyap_n_warmup": int(warmup_used),
        "lyap_k": int(diag["spectrum"].numel()),
        "lyap_map": amap.kind,
        "lyap_autonomous_input": AUTONOMOUS_INPUT,
    }


def attach_lyapunov_metrics(
    metrics: MutableMapping[str, Any],
    model,
    batch_data: Optional[torch.Tensor] = None,
    *,
    enabled: bool = False,
    n_steps: int = DEFAULT_N_STEPS,
    n_transient: int = DEFAULT_N_TRANSIENT,
    n_warmup: int = DEFAULT_N_WARMUP,
    k: Optional[int] = None,
    batch_index: int = 0,
) -> MutableMapping[str, Any]:
    """Write Lyapunov keys into ``metrics`` when ``enabled``.

    ``enabled=False`` returns ``metrics`` untouched (no model call).
    Unsupported architectures and non-finite free-runs set
    ``lyapunov_skip_reason`` and warn, instead of failing the eval.
    """
    if not enabled:
        return metrics
    try:
        result = compute_lyapunov_metrics(
            model,
            batch_data,
            n_steps=n_steps,
            n_transient=n_transient,
            n_warmup=n_warmup,
            k=k,
            batch_index=batch_index,
        )
    except (UnsupportedLatentMap, RuntimeError, ValueError) as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        metrics["lyapunov_skip_reason"] = str(exc)
        return metrics
    metrics.update(result)
    return metrics


# Imported for symmetry with the cell tests; the eval path does not call it.
def clipped_shplrnn_step(
    z: torch.Tensor,
    A: torch.Tensor,
    W1: torch.Tensor,
    W2: torch.Tensor,
    b2: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Durstewitz clipped shallow step: W1 (φ(W2z+b2) - φ(W2z)) + A⊙z + b1.

    Not what ``shPLRNN.forward`` implements. Exposed so the clipped gate
    can be checked against autodiff without changing the trained cell.
    """
    pre = z @ W2.T + b2
    phi = F.relu(pre) - F.relu(z @ W2.T)
    out = A * z + phi @ W1.T
    if b1 is not None:
        out = out + b1
    return out
