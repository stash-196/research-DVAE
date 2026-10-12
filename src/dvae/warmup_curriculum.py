"""Warm-up phase order and the patience that triggers each bump.

Two `[Training]` keys, both defaulting to today's loop:

* `warmup_order = auto_first` (blank does the same). Autonomous and noise
  bump together, then KL, then sequence length. `window_first` grows the
  sequence to `sequence_len` first, while the autonomous fraction stays at
  `auto_warm_start` (full teacher forcing when that start is 0), then
  autonomous and noise, then KL.
* `warmup_patience_start` blank, none, null, or off: every bump and the
  final stop use `early_stop_patience`. When set, the patience that fires
  each bump grows linearly from that start to `early_stop_patience` across
  the planned bumps. The final stop, once every phase is done, always uses
  the full `early_stop_patience`.

Planned bumps are counted once from the config: window steps + autonomous
steps + KL steps. A sequence-length bump still resets KL and the autonomous
fraction the way the training loop always has, so `auto_first` can fire
more bump events than that sum. Those extra events sit at the end of the
schedule and use `early_stop_patience`.

Step counts replay the live bump arithmetic, including the `0.1` float
residue (KL and a 0-to-limit autonomous ramp each take 11 steps, not 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WARMUP_BUMP = 0.1
ORDER_AUTO_FIRST = "auto_first"
ORDER_WINDOW_FIRST = "window_first"
_ORDERS = (ORDER_AUTO_FIRST, ORDER_WINDOW_FIRST)
_PATIENCE_OFF = {"", "none", "null", "off", "false", "no"}


def resolve_warmup_order(value) -> str:
    """Blank or ``auto_first`` keeps the historical phase order."""
    if value is None:
        return ORDER_AUTO_FIRST
    key = str(value).strip().lower()
    if key in {"", "none", "null", ORDER_AUTO_FIRST}:
        return ORDER_AUTO_FIRST
    if key == ORDER_WINDOW_FIRST:
        return ORDER_WINDOW_FIRST
    raise ValueError(
        "warmup_order must be auto_first or window_first, "
        f"got {value!r}."
    )


def resolve_warmup_patience_start(value) -> float | None:
    """Blank, none, null, or off leaves bump patience at early_stop_patience."""
    if value is None or isinstance(value, bool):
        if value is True:
            raise ValueError(
                "warmup_patience_start must be a non-negative number, or "
                "blank to leave it off."
            )
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().lower()
        if text in _PATIENCE_OFF:
            return None
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(
                "warmup_patience_start must be a non-negative number, or "
                f"blank/none/off, got {value!r}."
            ) from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(
            "warmup_patience_start must be a non-negative finite number, "
            f"got {value!r}."
        )
    return number


def warmup_settings_from_config(cfg) -> tuple[str, float | None]:
    """Read the two keys. Missing options match a blank template."""
    order_raw = None
    patience_raw = None
    if cfg.has_section("Training"):
        order_raw = cfg.get("Training", "warmup_order", fallback=None)
        patience_raw = cfg.get("Training", "warmup_patience_start", fallback=None)
    return (
        resolve_warmup_order(order_raw),
        resolve_warmup_patience_start(patience_raw),
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


def patience_at_bump(
    bump_index: int,
    n_planned: int,
    patience_start: float,
    early_stop_patience: float,
) -> float:
    """Patience that must be exceeded to fire planned bump `bump_index`.

    Index 0 is the first bump and uses `patience_start`. The last planned
    bump (index `n_planned - 1`) uses `early_stop_patience`. One planned
    bump uses the start; the final stop is a separate check and always
    uses the full patience. Indexes past the schedule stay at the full
    patience (the extra `auto_first` bumps after a window reset).
    """
    start = float(patience_start)
    early = float(early_stop_patience)
    if n_planned <= 0:
        return early
    if n_planned == 1:
        return start if bump_index <= 0 else early
    if bump_index >= n_planned - 1:
        return early
    index = max(int(bump_index), 0)
    return start + (early - start) * (index / (n_planned - 1))


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


class WarmupCurriculum:
    """Phase order, planned-bump count, and one application of a bump."""

    def __init__(
        self,
        *,
        order=ORDER_AUTO_FIRST,
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
        self.n_planned = self.window_steps + self.autonomous_steps + self.kl_steps

    def finished(self, auto, noise, kl, sequence_len) -> bool:
        kl_done = (not self.enable_kl) or kl >= 1.0
        auto_done = (not self.enable_autonomous) or auto >= self.auto_limit
        noise_done = (not self.enable_noise) or noise >= self.noise_limit
        seq_done = sequence_len >= self.sequence_len
        return kl_done and auto_done and noise_done and seq_done

    def active_patience(self, bumps_completed: int, warmups_finished: bool) -> float:
        """Threshold compared with `cpt_patience` on this epoch.

        Finished warm-ups, a blank start, or nothing planned: the full
        `early_stop_patience`. Otherwise the linear value for the next bump.
        """
        early = float(self.early_stop_patience)
        if (
            warmups_finished
            or self.patience_start is None
            or self.n_planned <= 0
        ):
            return early
        return patience_at_bump(
            bumps_completed,
            self.n_planned,
            self.patience_start,
            self.early_stop_patience,
        )

    def apply_bump(self, auto, noise, kl, sequence_len) -> BumpResult:
        """One bump. Window bumps reset KL, auto, and (optionally) noise."""
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
        kind = "none"
        for phase in phases:
            if ready[phase]:
                kind = phase
                break
        auto_bumped = False
        noise_bumped = False
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
            if self.enable_kl:
                kl = 0.0
            auto = self.auto_start
            if self.noise_reset_on_window and self.enable_noise:
                noise = self.noise_start
        return BumpResult(
            kind=kind,
            auto=auto,
            noise=noise,
            kl=kl,
            sequence_len=sequence_len,
            auto_bumped=auto_bumped,
            noise_bumped=noise_bumped,
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
    bumps_completed = 0
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
        active = curriculum.active_patience(bumps_completed, finished)
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
                }
            )
            patience = 0
            warm_up_happened = True
            if bumped.kind != "none":
                bumps_completed += 1
    return Simulation(events, active_each, False)
