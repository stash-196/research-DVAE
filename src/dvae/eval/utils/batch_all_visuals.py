"""
Cross-batch (batch_all) visualizations for multi-start metrics.

Scalars stay mean±std over independent windows. These figures only stitch
signals for illustration: NaN gaps between batches avoid artificial jumps in
delay embeddings and series plots.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence  # Sequence used by concat + refs

import numpy as np

from dvae.eval.utils.delay_embedding import compute_delay_embedding
from dvae.eval.utils.durstewitz_eval_metrics import state_space_kl
from dvae.eval.utils.frequency_analysis import _spectrum_pair_errors
from dvae.visualizers import visualize_delay_embedding
from dvae.visualizers.visualizers import (
    visualize_errors_from_lst,
    visualize_sequences,
)


def _as_1d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64).reshape(-1)
    return arr


def concat_with_nan_gaps(
    segments: Sequence[np.ndarray], gap: int = 1
) -> np.ndarray:
    """Concatenate 1D segments with ``gap`` NaNs between each pair."""
    pieces: List[np.ndarray] = []
    nan_pad = np.full(max(0, int(gap)), np.nan, dtype=np.float64)
    for i, seg in enumerate(segments):
        pieces.append(_as_1d(seg))
        if i < len(segments) - 1 and gap > 0:
            pieces.append(nan_pad)
    if not pieces:
        return np.array([], dtype=np.float64)
    return np.concatenate(pieces)


def _primary_channel(channel_benchmarks: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    chans = channel_benchmarks.get("channels") or []
    if not chans:
        return None
    # Prefer explicit x / signal / first channel
    for prefer in ("x", "signal", "ch4", "ch1"):
        for ch in chans:
            if ch.get("key") == prefer:
                return ch
    return chans[0]


def collect_batch_visual_record(
    channel_benchmarks: Dict[str, Any],
    recon_mixed_full: np.ndarray,
    mode_selector_1d: np.ndarray,
    batch_idx: int,
    start_frame: int,
) -> Optional[Dict[str, Any]]:
    """
    Snapshot one metrics window for later batch_all stitching.

    recon_mixed_full: (seq_len,) reconstruction under the metrics drive schedule
    mode_selector_1d: (seq_len,) 0=TF, 1=Auto
    """
    ch = _primary_channel(channel_benchmarks)
    if ch is None:
        return None
    recon = _as_1d(recon_mixed_full)
    mode = _as_1d(mode_selector_1d)
    gt_full = _as_1d(ch["gt_full"])
    n = min(len(gt_full), len(recon), len(mode))
    return {
        "batch_idx": batch_idx,
        "start_frame": int(start_frame),
        "key": ch["key"],
        "gt_full": gt_full[:n],
        "recon_mixed": recon[:n],
        "mode_1d": mode[:n],
        "gt_auto": _as_1d(ch["gt_auto"]),
        "tf_auto": _as_1d(ch["tf_auto"]),
        "auto_seg": _as_1d(ch["auto_seg"]),
        "time_delay": channel_benchmarks.get("time_delay", 10),
        "delay_dims": channel_benchmarks.get("delay_dims", 3),
    }


def render_batch_all_visuals(
    records: List[Dict[str, Any]],
    save_dir: str,
    gap: int = 1,
    explain_suffix: str = "",
) -> None:
    """
    Write batch_all series + delay-embedding GIFs under ``save_dir``.

    - Series: full-window GT + mixed recon colored by mode, NaN-separated batches
    - Delay embeds: GT / TF / Auto free-run segments only, NaN-separated across batches
    """
    if not records:
        print("[Eval] batch_all: no records to visualize.")
        return

    os.makedirs(save_dir, exist_ok=True)
    n_batches = len(records)
    key = records[0].get("key", "x")
    time_delay = int(records[0].get("time_delay", 10))
    delay_dims = int(records[0].get("delay_dims", 3))
    tag = explain_suffix or f"n{n_batches}"

    # --- Series (full drive schedule per window) ---
    gt_series = concat_with_nan_gaps([r["gt_full"] for r in records], gap=gap)
    recon_series = concat_with_nan_gaps([r["recon_mixed"] for r in records], gap=gap)
    mode_series = concat_with_nan_gaps([r["mode_1d"] for r in records], gap=gap)
    # mode gaps: fill with 0 so plot code doesn't break; recon/gt already NaN
    mode_plot = np.nan_to_num(mode_series, nan=0.0)

    try:
        visualize_sequences(
            sequences=[
                {"data": gt_series, "name": "True"},
                {"data": recon_series, "name": "Recon"},
            ],
            mode_selector=mode_plot,
            save_dir=save_dir,
            explain=f"metrics_drive_batch_all_{tag}_mode_colored",
        )
        print(f"[Eval] batch_all series saved under {save_dir}")
    except Exception as exc:
        print(f"[Eval] batch_all series failed: {exc}")

    # --- Delay embeddings on free-run fragments only ---
    for kind, color, field in (
        ("gt", "Blues", "gt_auto"),
        ("tf", "Greens", "tf_auto"),
        ("auto", "Reds", "auto_seg"),
    ):
        stitched = concat_with_nan_gaps([r[field] for r in records], gap=gap)
        try:
            emb = compute_delay_embedding(
                stitched, delay=time_delay, dimensions=delay_dims, handle_nan="remove"
            )
            visualize_delay_embedding(
                embedded=emb,
                save_dir=save_dir,
                variable_name=f"{key}_{kind}_tau{time_delay}_d{delay_dims}",
                explain=f"batch_all_{tag}",
                base_color=color,
            )
            print(f"[Eval] batch_all delay embed ({kind}) saved under {save_dir}")
        except Exception as exc:
            print(f"[Eval] batch_all delay embed ({kind}) skipped: {exc}")

    # Meta note
    meta_path = os.path.join(save_dir, "batch_all_readme.txt")
    with open(meta_path, "w") as f:
        f.write(
            "batch_all visuals only (not used for YAML scalar metrics).\n"
            f"n_batches={n_batches}\n"
            f"nan_gap={gap} between windows so delay-embed does not bridge jumps.\n"
            "Series: full metrics window (TF+Auto schedule) per batch, concatenated.\n"
            "Delay embeds: free-run (Auto-segment) GT/TF/Auto only, concatenated.\n"
            f"batch start_frames: {[r.get('start_frame') for r in records]}\n"
        )


def _extract_full_multichannel(raw) -> tuple:
    """
    Normalize get_full_xyz output to (data 2D float array, column_names list|None).

    Supports torch tensors, ndarrays, and pandas DataFrames (XHRO).
    XHRO DataFrames include non-numeric cols (datetime, expt_id, ...); we keep
    preferred signal columns (ch1–ch4 / x,y,z) when present.
    """
    # pandas DataFrame
    if hasattr(raw, "to_numpy") and hasattr(raw, "columns"):
        cols_all = [str(c) for c in list(raw.columns)]
        prefer = ["ch1", "ch2", "ch3", "ch4", "x", "y", "z"]
        # exact preferred names first
        keep = [c for c in prefer if c in cols_all]
        if not keep:
            # fallback: any column that looks like chN or is numeric dtype
            keep = []
            for c in cols_all:
                cl = c.lower()
                if cl.startswith("ch") and cl[2:].isdigit():
                    keep.append(c)
            if not keep:
                # last resort: numeric dtypes only
                try:
                    num = raw.select_dtypes(include=["number"])
                    keep = [str(c) for c in num.columns]
                    raw = num
                    cols_all = keep
                except Exception:
                    keep = cols_all
        if keep and hasattr(raw, "loc"):
            # preserve prefer order among selected
            ordered = [c for c in prefer if c in keep] + [
                c for c in keep if c not in prefer
            ]
            sub = raw[ordered]
            data = sub.to_numpy(dtype=np.float64, copy=True)
            return data, ordered
        data = raw.to_numpy(dtype=np.float64, copy=True)
        return data, cols_all

    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    data = np.asarray(raw, dtype=np.float64)
    return data, None


def resolve_reference_channel_spec(
    dataset_name: str,
    observation_process: Optional[str] = None,
    primary_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Which physical channels to show as reference bars (vs the trained target).

    Lorenz: z, y vs x  (old paper order)
    XHRO:   ch1,ch2,ch3 vs target ch (usually ch4 / GT); if multi-ch train, all
            other ch* vs primary
    """
    name = (dataset_name or "").lower()
    if name == "lorenz63":
        return {
            "primary_name": "x",
            "primary_idx": 0,
            "refs": [("z", 2), ("y", 1)],  # bar order
            "column_names": ["x", "y", "z"],
        }
    if name in ("xhro", "xhropacketloss"):
        # Old eval: ch1, ch2, ch3, GroundTruth(ch4), TF, Auto
        all_ch = ["ch1", "ch2", "ch3", "ch4"]
        # Infer target channel from observation_process / primary_key
        target = None
        if primary_key and str(primary_key).lower() in all_ch:
            target = str(primary_key).lower()
        if target is None and observation_process:
            op = str(observation_process).lower()
            for c in all_ch:
                if c in op or op.endswith(c[-1]):  # raw_ch4, only ch4, etc.
                    if c in op:
                        target = c
                        break
            # common templates
            for c in all_ch:
                if op == c or op.startswith(c) or f"_{c}" in op or f"{c}_" in op:
                    target = c
                    break
            if "ch4" in op or op in (
                "only_x",
                "only_x_interpolate",
                "only_x_indicate",
                "raw_ch4",
            ):
                target = target or "ch4"
        target = target or "ch4"
        refs = [(c, all_ch.index(c)) for c in all_ch if c != target]
        # Old bar order was ch1, ch2, ch3 (natural index order among refs)
        refs = sorted(refs, key=lambda t: t[1])
        return {
            "primary_name": target,
            "primary_idx": all_ch.index(target),
            "refs": refs,
            "column_names": all_ch,
        }
    return None


def compute_reference_channel_errors(
    dataset,
    batch_indices: Sequence[int],
    auto_seg_len: int,
    dataset_name: str,
    observation_process: Optional[str] = None,
    primary_key: Optional[str] = None,
    time_delay: int = 10,
    delay_dims: int = 3,
    dt: float = 0.01,
    flip_point: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Paper-style baselines: other physical channels vs the trained target channel
    on the free-run window (MSE, delay-embed KLD, spectrum Hellinger).

    Lorenz bars: z | y | GT | TF | Auto
    XHRO bars:   ch1 | ch2 | ch3 | GT | TF | Auto  (when target is ch4)
    """
    if not hasattr(dataset, "get_full_xyz"):
        return {}
    if auto_seg_len < 4:
        return {}

    spec = resolve_reference_channel_spec(
        dataset_name, observation_process=observation_process, primary_key=primary_key
    )
    if not spec or not spec["refs"]:
        return {}

    primary_idx = int(spec["primary_idx"])
    refs = list(spec["refs"])  # list of (name, idx)
    sr = 1.0 / float(dt)
    acc: Dict[str, Dict[str, List[float]]] = {
        name: {"mse": [], "kld": [], "spectrum": []} for name, _ in refs
    }

    for bi in batch_indices:
        try:
            raw = dataset.get_full_xyz(int(bi))
            data, colnames = _extract_full_multichannel(raw)
            if data.ndim != 2 or data.shape[0] < 4:
                continue

            # Map names -> columns if DataFrame-style names exist
            def col_index(name: str, fallback: int) -> int:
                if colnames is not None:
                    # exact or case-insensitive / substring (e.g. raw_ch1)
                    for j, c in enumerate(colnames):
                        cl = c.lower()
                        if cl == name or cl.endswith(name) or name in cl:
                            return j
                return fallback

            p_i = col_index(spec["primary_name"], primary_idx)
            if p_i >= data.shape[1]:
                continue

            if flip_point is not None and 0 < int(flip_point) < data.shape[0]:
                sl = slice(int(flip_point), None)
            else:
                n = min(int(auto_seg_len), data.shape[0])
                sl = slice(data.shape[0] - n, data.shape[0])

            primary = np.asarray(data[sl, p_i], dtype=np.float64).reshape(-1)
            # Linear-interpolate NaNs in primary for fair spectrum/KLD baselines
            if np.any(~np.isfinite(primary)):
                idx = np.arange(len(primary))
                ok = np.isfinite(primary)
                if ok.sum() < 4:
                    continue
                primary = np.interp(idx, idx[ok], primary[ok])

            min_emb = max(4, (delay_dims - 1) * time_delay + 2)
            if len(primary) < min_emb:
                continue

            try:
                p_emb = compute_delay_embedding(
                    primary, delay=time_delay, dimensions=delay_dims, handle_nan="remove"
                )
            except Exception:
                p_emb = None

            for ref_name, ref_fallback_idx in refs:
                r_i = col_index(ref_name, ref_fallback_idx)
                if r_i >= data.shape[1]:
                    continue
                other = np.asarray(data[sl, r_i], dtype=np.float64).reshape(-1)
                m = min(len(primary), len(other))
                if m < min_emb:
                    continue
                p = primary[:m]
                o = other[:m]
                if np.any(~np.isfinite(o)):
                    idx = np.arange(m)
                    ok = np.isfinite(o)
                    if ok.sum() < 4:
                        continue
                    o = np.interp(idx, idx[ok], o[ok])

                # MSE of other channel vs target channel
                acc[ref_name]["mse"].append(float(np.nanmean((p - o) ** 2)))

                try:
                    acc[ref_name]["spectrum"].append(_spectrum_pair_errors(p, o, sr))
                except Exception:
                    pass

                if p_emb is not None:
                    try:
                        o_emb = compute_delay_embedding(
                            o, delay=time_delay, dimensions=delay_dims, handle_nan="remove"
                        )
                        acc[ref_name]["kld"].append(
                            float(state_space_kl(p_emb, o_emb, use_gmm=True))
                        )
                    except Exception:
                        pass
        except Exception:
            continue

    out: Dict[str, Dict[str, float]] = {}
    # Preserve declared bar order
    for name, _ in refs:
        vals = {}
        for kind, lst in acc[name].items():
            finite = [v for v in lst if np.isfinite(v)]
            if finite:
                vals[kind] = float(np.mean(finite))
        if vals:
            out[name] = vals
    return out


# Back-compat alias
def compute_lorenz_reference_errors(*args, **kwargs):
    kwargs.setdefault("dataset_name", "Lorenz63")
    return compute_reference_channel_errors(*args, **kwargs)


def _finite_stats(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "iqr": float("nan"),
            "values": [],
        }
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "values": [float(v) for v in arr],
    }


def compute_stitched_kld_metrics(
    records: List[Dict[str, Any]],
    geom_results_list: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    KLD for batch_all reporting:

    - **stitched**: one GMM/KL on pooled free-run delay clouds (aligned with
      batch_all delay-embed GIFs / old even_bursts mask-gather style).
    - **per-window**: median / IQR / mean of independent free-run KLDs (reliability).

    Primary bar height should use stitched; spread should use per-window IQR.
    """
    out: Dict[str, Any] = {
        "kld_tf_stitched": float("nan"),
        "kld_auto_stitched": float("nan"),
        "kld_tf_per_window": {},
        "kld_auto_per_window": {},
        "kld_metric_note": (
            "kld_*_stitched = KL on pooled free-run delay-embed clouds across "
            "windows (geometry occupancy; matches batch_all GIF). "
            "kld_*_per_window_* = distribution of independent free-run KLDs "
            "(median/IQR preferred over mean when heavy-tailed)."
        ),
    }
    if not records:
        return out

    td = int(records[0].get("time_delay", 10))
    dd = int(records[0].get("delay_dims", 3))

    def _kld_pair(gt_seg, gen_seg) -> float:
        gt_e = compute_delay_embedding(
            np.asarray(gt_seg, dtype=np.float64).reshape(-1),
            delay=td,
            dimensions=dd,
            handle_nan="remove",
        )
        gen_e = compute_delay_embedding(
            np.asarray(gen_seg, dtype=np.float64).reshape(-1),
            delay=td,
            dimensions=dd,
            handle_nan="remove",
        )
        return float(state_space_kl(gt_e, gen_e, use_gmm=True))

    # --- stitched (pooled occupancy) ---
    try:
        gt_cat = concat_with_nan_gaps([r["gt_auto"] for r in records], gap=1)
        tf_cat = concat_with_nan_gaps([r["tf_auto"] for r in records], gap=1)
        auto_cat = concat_with_nan_gaps([r["auto_seg"] for r in records], gap=1)
        out["kld_tf_stitched"] = _kld_pair(gt_cat, tf_cat)
        out["kld_auto_stitched"] = _kld_pair(gt_cat, auto_cat)
    except Exception as exc:
        print(f"[Eval] stitched KLD failed: {exc}")

    # --- per-window distribution (from geom results if available, else recompute) ---
    tf_ws: List[float] = []
    auto_ws: List[float] = []
    if geom_results_list:
        for g in geom_results_list:
            if not g:
                continue
            if "kld_tf" in g and np.isfinite(g["kld_tf"]):
                tf_ws.append(float(g["kld_tf"]))
            if "kld_auto" in g and np.isfinite(g["kld_auto"]):
                auto_ws.append(float(g["kld_auto"]))
    if len(auto_ws) < len(records):
        # recompute from records for robustness
        tf_ws, auto_ws = [], []
        for r in records:
            try:
                tf_ws.append(_kld_pair(r["gt_auto"], r["tf_auto"]))
                auto_ws.append(_kld_pair(r["gt_auto"], r["auto_seg"]))
            except Exception:
                continue

    out["kld_tf_per_window"] = _finite_stats(tf_ws)
    out["kld_auto_per_window"] = _finite_stats(auto_ws)

    # Convenience top-level keys for YAML aggregation / primary reporting
    out["kld_tf"] = out["kld_tf_stitched"]
    out["kld_auto"] = out["kld_auto_stitched"]
    out["kld_tf_median"] = out["kld_tf_per_window"].get("median", float("nan"))
    out["kld_auto_median"] = out["kld_auto_per_window"].get("median", float("nan"))
    out["kld_tf_iqr"] = out["kld_tf_per_window"].get("iqr", float("nan"))
    out["kld_auto_iqr"] = out["kld_auto_per_window"].get("iqr", float("nan"))
    out["kld_tf_q25"] = out["kld_tf_per_window"].get("q25", float("nan"))
    out["kld_tf_q75"] = out["kld_tf_per_window"].get("q75", float("nan"))
    out["kld_auto_q25"] = out["kld_auto_per_window"].get("q25", float("nan"))
    out["kld_auto_q75"] = out["kld_auto_per_window"].get("q75", float("nan"))
    out["kld_tf_mean_across_windows"] = out["kld_tf_per_window"].get(
        "mean", float("nan")
    )
    out["kld_auto_mean_across_windows"] = out["kld_auto_per_window"].get(
        "mean", float("nan")
    )
    out["kld_tf_std_across_windows"] = out["kld_tf_per_window"].get("std", float("nan"))
    out["kld_auto_std_across_windows"] = out["kld_auto_per_window"].get(
        "std", float("nan")
    )
    return out


def render_summary_error_bars(
    mse_results_list: List[Dict[str, Any]],
    geom_results_list: List[Dict[str, Any]],
    spectrum_results_list: List[Dict[str, Any]],
    save_dir: str,
    reference_errors: Optional[Dict[str, Dict[str, float]]] = None,
    kld_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Old-style multi-bar summary: optional refs, then GT(0), TF, Auto.

    For KLD, if ``kld_metrics`` is provided (from ``compute_stitched_kld_metrics``):
      - bar height = stitched-cloud KLD
      - black diamond = per-window median
      - whiskers = IQR of per-window KLD (around the median)
    """
    os.makedirs(save_dir, exist_ok=True)

    def _mean_key(results_list, key_tf, key_auto):
        tfs, autos = [], []
        for r in results_list:
            if r is None:
                continue
            if key_tf in r and np.isfinite(r[key_tf]):
                tfs.append(float(r[key_tf]))
            if key_auto in r and np.isfinite(r[key_auto]):
                autos.append(float(r[key_auto]))
        return (
            float(np.mean(tfs)) if tfs else float("nan"),
            float(np.mean(autos)) if autos else float("nan"),
            float(np.std(tfs)) if len(tfs) > 1 else 0.0,
            float(np.std(autos)) if len(autos) > 1 else 0.0,
        )

    # KLD bar heights: prefer stitched
    if kld_metrics is not None:
        kld_tf_bar = float(kld_metrics.get("kld_tf_stitched", float("nan")))
        kld_auto_bar = float(kld_metrics.get("kld_auto_stitched", float("nan")))
        kld_tf_s = float(kld_metrics.get("kld_tf_std_across_windows", 0.0) or 0.0)
        kld_auto_s = float(kld_metrics.get("kld_auto_std_across_windows", 0.0) or 0.0)
    else:
        kld_tf_bar, kld_auto_bar, kld_tf_s, kld_auto_s = _mean_key(
            geom_results_list, "kld_tf", "kld_auto"
        )

    specs = [
        (
            "mse_error_per_signal",
            "MSE",
            _mean_key(mse_results_list, "mse_tf", "mse_auto"),
            "mse",
            False,
        ),
        (
            "kld_error_per_signal",
            "KLD",
            (kld_tf_bar, kld_auto_bar, kld_tf_s, kld_auto_s),
            "kld",
            True,  # use stitched + median/IQR annotations
        ),
        (
            "power_spectrum_error",
            "dB",
            _mean_key(
                spectrum_results_list, "spectrum_error_tf", "spectrum_error_auto"
            ),
            "spectrum",
            False,
        ),
    ]

    ref_color = {
        "z": "orange",
        "y": "magenta",
        "ch1": "cyan",
        "ch2": "orange",
        "ch3": "magenta",
    }

    for explain, unit, (tf_m, auto_m, tf_s, auto_s), ref_kind, is_kld in specs:
        names: List[str] = []
        errs: List[float] = []
        colors: List[str] = []
        yerr: List[Any] = []
        medians: List[Any] = []

        if reference_errors:
            for ref_name in reference_errors:
                ref_vals = reference_errors[ref_name]
                if ref_kind in ref_vals and np.isfinite(ref_vals[ref_kind]):
                    names.append(ref_name)
                    errs.append(float(ref_vals[ref_kind]))
                    colors.append(ref_color.get(ref_name, "gray"))
                    yerr.append(None)
                    medians.append(None)

        names.append("Ground\nTruth")
        errs.append(0.0)
        colors.append("blue")
        yerr.append(None)
        medians.append(None)

        names.append("Teacher-\nForced")
        errs.append(tf_m)
        colors.append("green")
        names.append("Autonomous")
        errs.append(auto_m)
        colors.append("red")

        subtitle = None
        if is_kld and kld_metrics is not None:
            # IQR whiskers around per-window median (not around stitched bar)
            # Visualize whiskers relative to bar height? Better: yerr from median
            # with diamond at median; bar stays stitched.
            def _iqr_yerr(mode: str):
                q25 = kld_metrics.get(f"kld_{mode}_q25", float("nan"))
                med = kld_metrics.get(f"kld_{mode}_median", float("nan"))
                q75 = kld_metrics.get(f"kld_{mode}_q75", float("nan"))
                stitched = kld_metrics.get(f"kld_{mode}_stitched", float("nan"))
                if not all(np.isfinite(v) for v in (q25, med, q75, stitched)):
                    return None, None
                # Whiskers drawn from bar (stitched): distance stitched->q25/q75
                # can go negative; use max(0, ...) for matplotlib yerr
                # Better approach: set yerr around median by temporarily using
                # errorbar only; here pass IQR half-widths relative to stitched
                # so whisker ends land on q25/q75 when bar is at stitched:
                # low = stitched - q25 (can be negative → clamp display via abs)
                # User asked: bar=stitched, show median and spread.
                # We'll put yerr so whiskers span [q25, q75] by using
                # err_low = max(0, stitched - q25), err_high = max(0, q75 - stitched)
                # if stitched outside IQR, one side zero — OK.
                return (
                    max(0.0, float(stitched - q25)),
                    max(0.0, float(q75 - stitched)),
                ), float(med)

            tf_ye, tf_med = _iqr_yerr("tf")
            auto_ye, auto_med = _iqr_yerr("auto")
            yerr.extend([tf_ye, auto_ye])
            medians.extend([tf_med, auto_med])
            subtitle = (
                "Bars = stitched free-run cloud KLD (batch_all geometry). "
                "Diamonds = per-window median; whiskers span per-window IQR."
            )
        else:
            yerr.extend([None, None])
            medians.extend([None, None])

        if not any(np.isfinite(e) for e in errs):
            continue
        errs_plot = [0.0 if not np.isfinite(e) else e for e in errs]
        visualize_errors_from_lst(
            errs_plot,
            name_lst=names,
            save_dir=save_dir,
            explain=explain,
            error_unit=unit,
            colors=colors,
            yerr=yerr if is_kld and kld_metrics is not None else None,
            median_markers=medians if is_kld and kld_metrics is not None else None,
            subtitle=subtitle,
        )
        if is_kld and kld_metrics is not None:
            print(
                f"[Eval] batch_all KLD bars (stitched): "
                f"TF={kld_metrics.get('kld_tf_stitched')}  "
                f"Auto={kld_metrics.get('kld_auto_stitched')}  |  "
                f"median TF={kld_metrics.get('kld_tf_median')} "
                f"IQR=[{kld_metrics.get('kld_tf_q25')}, {kld_metrics.get('kld_tf_q75')}]  "
                f"median Auto={kld_metrics.get('kld_auto_median')} "
                f"IQR=[{kld_metrics.get('kld_auto_q25')}, {kld_metrics.get('kld_auto_q75')}]"
            )
        else:
            print(
                f"[Eval] batch_all summary bars {explain}: "
                f"TF={tf_m:.4g}±{tf_s:.4g}, Auto={auto_m:.4g}±{auto_s:.4g}"
            )
