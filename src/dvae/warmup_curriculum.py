"""Warm-up phase order and the patience that triggers each bump.

Two `[Training]` keys. Blank keys take the new-run defaults:

* `warmup_order` blank is `window_first`. The sequence grows to
  `sequence_len` first, while the autonomous fraction stays at
  `auto_warm_start` (full teacher forcing when that start is 0). A
  window bump does not reset autonomous, noise, or KL. Those ramps then
  run once, at the full length. `auto_first` is the previous nested
  order: a full autonomous ramp, then KL, at every window length, and
  each sequence-length bump resets autonomous, KL, and noise.
* `warmup_patience_start` blank is 2. In ``window_first`` every
  sequence-length bump uses that start. Patience then rises only over
  the autonomous and KL bumps:
  ``start + (early_stop_patience - start) * k / K``, where ``k`` is how
  many of those bumps have already fired and ``K`` is how many are
  planned. The final stop uses the full ``early_stop_patience``. With
  no autonomous or KL bumps, window bumps stay at the start and the
  stop uses the full patience. ``auto_first`` keeps a single patience,
  ``early_stop_patience``, for every bump and the stop. ``off`` does
  the same in ``window_first``.

`[DataFrame] legacy_defaults` true, or the old name `legacy_windowing`,
restores `auto_first` and the single patience when those two keys are
blank. An explicit order or an explicit patience still wins.

Bump counts, separate from that patience length: ``window_first`` is
window bumps plus one autonomous ramp plus KL. ``auto_first`` is the
nested total ``window_steps * inner + window_steps + inner``, where
``inner`` is autonomous bumps plus KL bumps.

Step counts replay the live bump arithmetic, including the `0.1` float
residue (KL and a 0-to-limit autonomous ramp each take 11 steps, not 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WARMUP_BUMP = 0.1
ORDER_AUTO_FIRST = "auto_first"
ORDER_WINDOW_FIRST = "window_first"
DEFAULT_WARMUP_ORDER = ORDER_WINDOW_FIRST
DEFAULT_WARMUP_PATIENCE_START = 2.0
_ORDERS = (ORDER_AUTO_FIRST, ORDER_WINDOW_FIRST)
_UNSET = {"", "none", "null"}
_PATIENCE_OFF = {"off", "false", "no"}


def resolve_warmup_order(value, legacy: bool = False) -> str:
    """Blank is ``window_first``. Legacy blank is ``auto_first``.

    An explicit ``auto_first`` or ``window_first`` wins over ``legacy``.
    """
    if value is None:
        key = ""
    else:
        key = str(value).strip().lower()
    if key in _UNSET:
        if legacy:
            return ORDER_AUTO_FIRST
        return DEFAULT_WARMUP_ORDER
    if key == ORDER_AUTO_FIRST:
        return ORDER_AUTO_FIRST
    if key == ORDER_WINDOW_FIRST:
        return ORDER_WINDOW_FIRST
    raise ValueError(
        "warmup_order must be auto_first or window_first, "
        f"got {value!r}."
    )


def resolve_warmup_patience_start(value, legacy: bool = False) -> float | None:
    """Blank is 2. ``off`` is the single ``early_stop_patience``.

    Legacy blank is that single patience too. A number is that start,
    including under legacy. ``None`` here is a missing config value, not
    the already-resolved "patience is off" stored on ``WarmupCurriculum``.
    """
    if isinstance(value, bool):
        if value is True:
            raise ValueError(
                "warmup_patience_start must be a non-negative number, "
                "or off for a single early_stop_patience."
            )
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = "" if value is None else str(value).strip().lower()
        if text in _UNSET:
            if legacy:
                return None
            return DEFAULT_WARMUP_PATIENCE_START
        if text in _PATIENCE_OFF:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(
                "warmup_patience_start must be a non-negative number, "
                f"or off, got {value!r}."
            ) from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            "warmup_patience_start must be a non-negative finite number, "
            f"got {value!r}."
        )
    return number


def warmup_settings_from_config(cfg) -> tuple[str, float | None]:
    """Read the two keys. Missing options match a blank template.

    Blank is ``window_first`` and patience 2. ``legacy_defaults`` or
    ``legacy_windowing`` changes those blanks to ``auto_first`` and a
    single patience. Explicit values still win.
    """
    from dvae.run_options import legacy_defaults_from_config

    legacy = legacy_defaults_from_config(cfg)
    order_raw = None
    patience_raw = None
    if cfg.has_section("Training"):
        order_raw = cfg.get("Training", "warmup_order", fallback=None)
        patience_raw = cfg.get("Training", "warmup_patience_start", fallback=None)
    return (
        resolve_warmup_order(order_raw, legacy=legacy),
        resolve_warmup_patience_start(patience_raw, legacy=legacy),
    )


def format_patience(value) -> str:
    """Log whole-number patience as an int so the default line stays `10`."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def window_step_count(
    initial_sequence_len: int,
    sequence_len: int,
    bump: float = WARMUP_BUMP,
) -> int:
    """How many `+ int(bump * sequence_len)` updates reach the target."""
    current = int(initial_sequence_len)
    target = int(sequence_len)
    step = int(bump * target)
    steps = 0
    if current >= target or step <= 0:
        return 0
    while current < target:
        nxt = min(target, current + step)
        if nxt <= current:
            break
        current = nxt
        steps += 1
    return steps


def kl_step_count(enabled: bool, bump: float = WARMUP_BUMP, ceiling: float = 1.0) -> int:
    """KL bumps from 0 by `bump` until the value is no longer below 1."""
    if not enabled:
        return 0
    current = 0.0
    steps = 0
    while current < ceiling:
        nxt = current + bump
        if nxt <= current:
            break
        current = nxt
        steps += 1
        if steps > 100000:
            break
    return steps


def _advance_auto_noise(
    auto: float,
    noise: float,
    *,
    enable_noise: bool,
    auto_limit: float,
    noise_limit: float,
    bump: float,
) -> tuple[float, float, bool, bool]:
    auto_bumped = False
    noise_bumped = False
    if auto < auto_limit:
        auto = min(auto_limit, auto + bump * auto_limit)
        auto_bumped = True
    if enable_noise and noise < noise_limit:
        noise = min(noise_limit, noise + bump * noise_limit)
        noise_bumped = True
    return auto, noise, auto_bumped, noise_bumped


def autonomous_step_count(
    *,
    enable_autonomous: bool,
    auto_start: float,
    auto_limit: float,
    enable_noise: bool,
    noise_start: float,
    noise_limit: float,
    bump: float = WARMUP_BUMP,
) -> int:
    """Times the joint autonomous+noise branch runs before both are at limit.

    Noise only moves inside that branch, and the branch is skipped when
    autonomous warm-up is off. When both are enabled the count is how many
    shared triggers it takes for the slower of the two to finish.
    """
    if not enable_autonomous:
        return 0
    auto = float(auto_start)
    noise = float(noise_start)
    steps = 0
    while auto < auto_limit or (enable_noise and noise < noise_limit):
        new_auto, new_noise, auto_bumped, noise_bumped = _advance_auto_noise(
            auto,
            noise,
            enable_noise=enable_noise,
            auto_limit=auto_limit,
            noise_limit=noise_limit,
            bump=bump,
        )
        if not auto_bumped and not noise_bumped:
            break
        auto, noise = new_auto, new_noise
        steps += 1
        if steps > 100000:
            break
    return steps


def planned_bump_count(
    order: str,
    window_steps: int,
    autonomous_steps: int,
    kl_steps: int,
) -> int:
    """How many bump triggers the patience schedule stretches across.

    ``window_first`` runs each phase once: windows at the start values,
    then one autonomous ramp, then KL if it is enabled.

    ``auto_first`` repeats the inner ramp after every window reset,
    including one ramp at the final length:
    ``window_steps * inner + window_steps + inner``, with
    ``inner = autonomous_steps + kl_steps``.
    """
    inner = int(autonomous_steps) + int(kl_steps)
    windows = int(window_steps)
    if resolve_warmup_order(order) == ORDER_WINDOW_FIRST:
        return windows + inner
    return windows * inner + windows + inner


def auto_kl_patience(
    auto_kl_done: int,
    auto_kl_planned: int,
    patience_start: float,
    early_stop_patience: float,
) -> float:
    """Patience for the next autonomous or KL bump.

    ``p_k = start + (early_stop_patience - start) * k / K``. ``k`` is
    how many autonomous or KL bumps have already fired. ``K`` is how
    many are planned. The last of those bumps stays short of
    ``early_stop_patience``. The final stop is a separate check.
    """
    start = float(patience_start)
    early = float(early_stop_patience)
    planned = int(auto_kl_planned)
    if planned <= 0:
        return start
    k = max(int(auto_kl_done), 0)
    return start + (early - start) * (k / planned)


def warmup_action(cpt_patience, active_patience, warmups_finished: bool) -> str:
    """`continue`, `bump`, or `stop`. The comparison stays strict `>`."""
    if cpt_patience > active_patience:
        if warmups_finished:
            return "stop"
        return "bump"
    return "continue"


@dataclass(frozen=True)
class BumpResult:
    kind: str
    auto: float
    noise: float
    kl: float
    sequence_len: int
    auto_bumped: bool
    noise_bumped: bool
    reset_warmups: bool = False


class WarmupCurriculum:
    """Phase order, planned-bump count, and one application of a bump."""

    def __init__(
        self,
        *,
        order=DEFAULT_WARMUP_ORDER,
        patience_start=None,
        early_stop_patience: int,
        initial_sequence_len: int,
        sequence_len: int,
        enable_autonomous: bool,
        auto_start: float,
        auto_limit: float,
        enable_noise: bool,
        noise_start: float,
        noise_limit: float,
        noise_reset_on_window: bool,
        enable_kl: bool,
        bump: float = WARMUP_BUMP,
    ):
        self.order = resolve_warmup_order(order)
        # None is already resolved: adaptive patience is off. The config
        # reader turns a blank key into 2 before it reaches this constructor.
        if patience_start is None:
            self.patience_start = None
        else:
            self.patience_start = resolve_warmup_patience_start(patience_start)
        self.early_stop_patience = early_stop_patience
        self.initial_sequence_len = int(initial_sequence_len)
        self.sequence_len = int(sequence_len)
        self.enable_autonomous = bool(enable_autonomous)
        self.auto_start = float(auto_start)
        self.auto_limit = float(auto_limit)
        self.enable_noise = bool(enable_noise)
        self.noise_start = float(noise_start)
        self.noise_limit = float(noise_limit)
        self.noise_reset_on_window = bool(noise_reset_on_window)
        self.enable_kl = bool(enable_kl)
        self.bump = float(bump)
        self.window_steps = window_step_count(
            self.initial_sequence_len, self.sequence_len, self.bump
        )
        self.autonomous_steps = autonomous_step_count(
            enable_autonomous=self.enable_autonomous,
            auto_start=self.auto_start,
            auto_limit=self.auto_limit,
            enable_noise=self.enable_noise,
            noise_start=self.noise_start,
            noise_limit=self.noise_limit,
            bump=self.bump,
        )
        self.kl_steps = kl_step_count(self.enable_kl, self.bump)
        self.n_planned = planned_bump_count(
            self.order,
            self.window_steps,
            self.autonomous_steps,
            self.kl_steps,
        )

    def finished(self, auto, noise, kl, sequence_len) -> bool:
        kl_done = (not self.enable_kl) or kl >= 1.0
        auto_done = (not self.enable_autonomous) or auto >= self.auto_limit
        noise_done = (not self.enable_noise) or noise >= self.noise_limit
        seq_done = sequence_len >= self.sequence_len
        return kl_done and auto_done and noise_done and seq_done

    def bump_kind(self, auto, noise, kl, sequence_len) -> str:
        """Which phase the next bump would advance, or ``none``."""
        auto = float(auto)
        noise = float(noise)
        kl = float(kl)
        sequence_len = int(sequence_len)
        window_ready = sequence_len < self.sequence_len
        auto_ready = self.enable_autonomous and (
            auto < self.auto_limit
            or (self.enable_noise and noise < self.noise_limit)
        )
        kl_ready = self.enable_kl and kl < 1.0
        if self.order == ORDER_WINDOW_FIRST:
            phases = ("window", "auto", "kl")
        else:
            phases = ("auto", "kl", "window")
        ready = {"window": window_ready, "auto": auto_ready, "kl": kl_ready}
        for phase in phases:
            if ready[phase]:
                return phase
        return "none"

    def active_patience(self, auto_kl_done, auto, noise, kl, sequence_len) -> float:
        """Threshold compared with ``cpt_patience`` on this epoch.

        ``auto_first``, a blank start, and the final stop all use
        ``early_stop_patience``. In ``window_first``, sequence-length
        bumps stay at ``warmup_patience_start``. Autonomous and KL bumps
        use ``auto_kl_patience`` with ``k`` equal to how many of those
        bumps have already fired.
        """
        early = float(self.early_stop_patience)
        if self.order != ORDER_WINDOW_FIRST or self.patience_start is None:
            return early
        if self.finished(auto, noise, kl, sequence_len):
            return early
        kind = self.bump_kind(auto, noise, kl, sequence_len)
        if kind in ("auto", "kl"):
            return auto_kl_patience(
                auto_kl_done,
                self.autonomous_steps + self.kl_steps,
                self.patience_start,
                self.early_stop_patience,
            )
        return float(self.patience_start)

    def apply_bump(self, auto, noise, kl, sequence_len) -> BumpResult:
        """One bump.

        ``auto_first`` window bumps reset KL and the autonomous fraction,
        and reset noise when ``noise_reset_on_window`` is set.
        ``window_first`` window bumps leave those values alone.
        """
        auto = float(auto)
        noise = float(noise)
        kl = float(kl)
        sequence_len = int(sequence_len)
        kind = self.bump_kind(auto, noise, kl, sequence_len)
        auto_bumped = False
        noise_bumped = False
        reset_warmups = False
        if kind == "auto":
            auto, noise, auto_bumped, noise_bumped = _advance_auto_noise(
                auto,
                noise,
                enable_noise=self.enable_noise,
                auto_limit=self.auto_limit,
                noise_limit=self.noise_limit,
                bump=self.bump,
            )
        elif kind == "kl":
            kl = kl + self.bump
        elif kind == "window":
            sequence_len = min(
                self.sequence_len,
                sequence_len + int(self.bump * self.sequence_len),
            )
            if self.order != ORDER_WINDOW_FIRST:
                if self.enable_kl:
                    kl = 0.0
                auto = self.auto_start
                if self.noise_reset_on_window and self.enable_noise:
                    noise = self.noise_start
                reset_warmups = True
        return BumpResult(
            kind=kind,
            auto=auto,
            noise=noise,
            kl=kl,
            sequence_len=sequence_len,
            auto_bumped=auto_bumped,
            noise_bumped=noise_bumped,
            reset_warmups=reset_warmups,
        )


@dataclass
class Simulation:
    """Stubbed recon-loss replay of the patience and bump decisions."""

    events: list
    active_patience: list
    stopped: bool


def simulate_recon_warmup(
    val_losses,
    curriculum: WarmupCurriculum,
    *,
    auto: float,
    noise: float,
    kl: float,
    sequence_len: int,
    delta_threshold: float = 1e-4,
) -> Simulation:
    """Replay bump decisions on a list of validation losses.

    Matches the recon early-stop path: best starts at +inf, an improvement
    is `loss - best < -delta_threshold`, and any other epoch increments
    patience. The epoch after a bump treats that epoch's loss as the new
    best before the delta is computed (`warm_up_happened`), so patience
    becomes 1 on a flat loss. No model, dataloader, or dataset is used.
    """
    best = float("inf")
    patience = 0
    warm_up_happened = False
    auto_kl_done = 0
    events = []
    active_each = []
    for epoch, loss in enumerate(val_losses):
        loss = float(loss)
        if warm_up_happened:
            best = loss
            warm_up_happened = False
        delta = loss - best
        if delta < -delta_threshold:
            patience = 0
            best = loss
        else:
            patience += 1
        finished = curriculum.finished(auto, noise, kl, sequence_len)
        active = curriculum.active_patience(
            auto_kl_done, auto, noise, kl, sequence_len
        )
        active_each.append(active)
        action = warmup_action(patience, active, finished)
        if action == "stop":
            events.append(
                {
                    "epoch": epoch,
                    "kind": "stop",
                    "active": active,
                    "patience": patience,
                    "auto_before": auto,
                    "noise_before": noise,
                    "kl_before": kl,
                    "sequence_before": sequence_len,
                    "auto": auto,
                    "noise": noise,
                    "kl": kl,
                    "sequence_len": sequence_len,
                    "auto_bumped": False,
                    "noise_bumped": False,
                    "reset_warmups": False,
                }
            )
            return Simulation(events, active_each, True)
        if action == "bump":
            before = (auto, noise, kl, sequence_len)
            bumped = curriculum.apply_bump(auto, noise, kl, sequence_len)
            auto = bumped.auto
            noise = bumped.noise
            kl = bumped.kl
            sequence_len = bumped.sequence_len
            events.append(
                {
                    "epoch": epoch,
                    "kind": bumped.kind,
                    "active": active,
                    "patience": patience,
                    "auto_before": before[0],
                    "noise_before": before[1],
                    "kl_before": before[2],
                    "sequence_before": before[3],
                    "auto": auto,
                    "noise": noise,
                    "kl": kl,
                    "sequence_len": sequence_len,
                    "auto_bumped": bumped.auto_bumped,
                    "noise_bumped": bumped.noise_bumped,
                    "reset_warmups": bumped.reset_warmups,
                }
            )
            patience = 0
            warm_up_happened = True
            if bumped.kind in ("auto", "kl"):
                auto_kl_done += 1
    return Simulation(events, active_each, False)
