import numpy as np
from dvae.eval.utils import compute_delay_embedding, state_space_kl
from dvae.eval.utils.intrinsic_dim import (
    ID_PER_CHANNEL_PREFIXES,
    delay_dim_for_key,
    id_metrics_for_clouds,
)
from dvae.visualizers import visualize_delay_embedding
from dvae.visualizers.visualizers import visualize_errors_from_lst

color_to_base = {
    "blue": "Blues",
    "green": "Greens",
    "red": "Reds",
    "orange": "Oranges",
    "magenta": "Purples",
    "cyan": "GnBu",
    "yellow": "YlOrBr",
}


def _embed_signal(sig, time_delay, delay_dims):
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 1 or len(sig) < (delay_dims - 1) * time_delay + 2:
        return None
    # Packet-loss / NaN-heavy segments can yield zero valid delay rows.
    # Treat as skipped channel (caller already maps None -> nan KLD).
    try:
        return compute_delay_embedding(sig, delay=time_delay, dimensions=delay_dims)
    except ValueError as exc:
        print(f"[Eval][Geometry] Skipping embedding ({exc})")
        return None


def _nanmean(values):
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _joint_embed(channels, field, time_delay, delay_dims):
    """Stack per-channel delay clouds when every channel has the same rows."""
    if len(channels) < 2:
        return None
    embs = []
    for ch in channels:
        emb = _embed_signal(ch[field], time_delay, delay_dims)
        if emb is None:
            return None
        embs.append(emb)
    n_rows = embs[0].shape[0]
    if any(emb.shape[0] != n_rows for emb in embs):
        return None
    return np.hstack(embs)


def run_geometry_analysis_from_benchmarks(
    channel_benchmarks, save_fig_dir, batch_idx=0, save_figures=True
):
    channels = channel_benchmarks["channels"]
    time_delay = channel_benchmarks["time_delay"]
    delay_dims = channel_benchmarks["delay_dims"]
    by_channel = channel_benchmarks.get("delay_dims_by_channel")

    per_channel = {}
    tf_scores = []
    auto_scores = []
    tf_keys = []
    auto_keys = []
    tf_names = []
    auto_names = []
    tf_colors = []
    auto_colors = []

    print("[Eval] KLD (State-Space via Delay Embedding, per channel):")
    for ch in channels:
        key = ch["key"]
        ch_dims = delay_dim_for_key(delay_dims, by_channel, key)
        gt_emb = _embed_signal(ch["gt_auto"], time_delay, ch_dims)
        tf_emb = _embed_signal(ch["tf_auto"], time_delay, ch_dims)
        auto_emb = _embed_signal(ch["auto_seg"], time_delay, ch_dims)
        # ID on the same clouds as DE→KLD (GT / TF / Auto share this channel's m*).
        id_metrics = id_metrics_for_clouds(gt_emb, tf_emb, auto_emb)

        if gt_emb is None:
            per_channel[key] = {
                "kld_tf": float("nan"),
                "kld_auto": float("nan"),
                "delay_dims": int(ch_dims),
                **id_metrics,
            }
            continue

        safe_name = key.replace(" ", "_").lower()
        base_color = color_to_base.get(ch["color"], "Blues")
        if save_figures:
            visualize_delay_embedding(
                embedded=gt_emb,
                save_dir=save_fig_dir,
                variable_name=f"{safe_name}_gt_tau{time_delay}_d{ch_dims}",
                explain=f"batch{batch_idx}_auto_segment",
                base_color=base_color,
            )

        kld_tf = float("nan")
        kld_auto = float("nan")

        def _safe_kld(a, b):
            # GMM needs enough samples; packet-loss NaN pruning can leave too few.
            min_n = 10
            if a is None or b is None or len(a) < min_n or len(b) < min_n:
                return float("nan")
            try:
                return float(state_space_kl(a, b, use_gmm=True))
            except Exception as exc:
                print(f"[Eval][Geometry] state_space_kl failed ({exc}); using nan")
                return float("nan")

        if tf_emb is not None:
            kld_tf = _safe_kld(gt_emb, tf_emb)
            if save_figures and np.isfinite(kld_tf):
                visualize_delay_embedding(
                    embedded=tf_emb,
                    save_dir=save_fig_dir,
                    variable_name=f"{safe_name}_tf_tau{time_delay}_d{ch_dims}",
                    explain=f"batch{batch_idx}_teacher_forced",
                    base_color="Greens",
                )
        if auto_emb is not None:
            kld_auto = _safe_kld(gt_emb, auto_emb)
            if save_figures and np.isfinite(kld_auto):
                visualize_delay_embedding(
                    embedded=auto_emb,
                    save_dir=save_fig_dir,
                    variable_name=f"{safe_name}_auto_tau{time_delay}_d{ch_dims}",
                    explain=f"batch{batch_idx}_autonomous",
                    base_color="Reds",
                )

        per_channel[key] = {
            "kld_tf": kld_tf,
            "kld_auto": kld_auto,
            "delay_dims": int(ch_dims),
            **id_metrics,
        }
        print(
            f"  {key} KLD TF: {kld_tf:.4f}  Auto: {kld_auto:.4f}  "
            f"(tau={time_delay}, m={ch_dims}, "
            f"TwoNN GT/TF/Auto="
            f"{id_metrics['id_twonn_gt']:.3f}/"
            f"{id_metrics['id_twonn_tf']:.3f}/"
            f"{id_metrics['id_twonn_auto']:.3f})"
        )

        tf_scores.append(kld_tf)
        auto_scores.append(kld_auto)
        tf_keys.append(f"{key}_tf")
        auto_keys.append(f"{key}_auto")
        tf_names.append(f"{key}\nTF")
        auto_names.append(f"{key}\nAuto")
        tf_colors.append("green")
        auto_colors.append("red")

    tf_values = [v["kld_tf"] for v in per_channel.values() if np.isfinite(v["kld_tf"])]
    auto_values = [
        v["kld_auto"] for v in per_channel.values() if np.isfinite(v["kld_auto"])
    ]
    kld_tf_mean = float(np.mean(tf_values)) if tf_values else float("nan")
    kld_auto_mean = float(np.mean(auto_values)) if auto_values else float("nan")

    if save_figures:
        comb_errs, comb_names, comb_colors = [0.0], ["Ground\nTruth"], ["blue"]
        for ch_key, vals in per_channel.items():
            comb_names.append(f"{ch_key}\nTF")
            comb_errs.append(vals["kld_tf"])
            comb_colors.append("green")
            comb_names.append(f"{ch_key}\nAuto")
            comb_errs.append(vals["kld_auto"])
            comb_colors.append("red")
        comb_errs = [0.0 if not np.isfinite(e) else e for e in comb_errs]
        visualize_errors_from_lst(
            comb_errs,
            name_lst=comb_names,
            save_dir=save_fig_dir,
            explain=f"kld_error_per_signal_batch{batch_idx}",
            error_unit="KLD",
            colors=comb_colors,
        )

    id_means = {}
    for prefix in ID_PER_CHANNEL_PREFIXES:
        id_means[prefix] = _nanmean(v.get(prefix, float("nan")) for v in per_channel.values())

    joint_m = channel_benchmarks.get("joint_delay_dims")
    if joint_m is None:
        joint_m = delay_dims
    joint_metrics = {}
    if len(channels) > 1:
        joint_clouds = id_metrics_for_clouds(
            _joint_embed(channels, "gt_auto", time_delay, int(joint_m)),
            _joint_embed(channels, "tf_auto", time_delay, int(joint_m)),
            _joint_embed(channels, "auto_seg", time_delay, int(joint_m)),
        )
        joint_metrics = {
            "id_pr_joint_gt": joint_clouds["id_pr_gt"],
            "id_pr_joint_tf": joint_clouds["id_pr_tf"],
            "id_pr_joint_auto": joint_clouds["id_pr_auto"],
            "id_twonn_joint_gt": joint_clouds["id_twonn_gt"],
            "id_twonn_joint_tf": joint_clouds["id_twonn_tf"],
            "id_twonn_joint_auto": joint_clouds["id_twonn_auto"],
            "joint_delay_dims": int(joint_m),
        }

    return {
        "per_channel": per_channel,
        "kld_tf_mean": kld_tf_mean,
        "kld_auto_mean": kld_auto_mean,
        "kld_tf": kld_tf_mean,
        "kld_auto": kld_auto_mean,
        "kld_scores": tf_scores + auto_scores,
        "signal_keys": tf_keys + auto_keys,
        "tf_keys": tf_keys,
        "auto_keys": auto_keys,
        **id_means,
        **joint_metrics,
    }


def run_geometry_analysis(
    test_dataloader,
    recon_data_long,
    save_fig_dir,
    i,
    autonomous_mode_selector_long,
    dataset_name,
    batch_data_long=None,
    channel_benchmarks=None,
    save_figures=True,
):
    if channel_benchmarks is not None:
        return run_geometry_analysis_from_benchmarks(
            channel_benchmarks, save_fig_dir, i, save_figures=save_figures
        )

    from dvae.eval.utils.benchmark_signals import get_benchmark_signals

    sig_info = get_benchmark_signals(
        dataset_name,
        test_dataloader,
        i,
        recon_data_long,
        autonomous_mode_selector_long,
        batch_data_long,
    )
    return run_geometry_analysis_from_benchmarks(
        sig_info["channel_benchmarks"], save_fig_dir, i, save_figures=save_figures
    )