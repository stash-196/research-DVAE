import numpy as np
from dvae.eval.utils import compute_delay_embedding, state_space_kl
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
    return compute_delay_embedding(sig, delay=time_delay, dimensions=delay_dims)


def run_geometry_analysis_from_benchmarks(
    channel_benchmarks, save_fig_dir, batch_idx=0, save_figures=True
):
    channels = channel_benchmarks["channels"]
    time_delay = channel_benchmarks["time_delay"]
    delay_dims = channel_benchmarks["delay_dims"]

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
        gt_emb = _embed_signal(ch["gt_auto"], time_delay, delay_dims)
        tf_emb = _embed_signal(ch["tf_auto"], time_delay, delay_dims)
        auto_emb = _embed_signal(ch["auto_seg"], time_delay, delay_dims)

        if gt_emb is None:
            per_channel[key] = {"kld_tf": float("nan"), "kld_auto": float("nan")}
            continue

        safe_name = key.replace(" ", "_").lower()
        base_color = color_to_base.get(ch["color"], "Blues")
        if save_figures:
            visualize_delay_embedding(
                embedded=gt_emb,
                save_dir=save_fig_dir,
                variable_name=f"{safe_name}_gt_tau{time_delay}_d{delay_dims}",
                explain=f"batch{batch_idx}_auto_segment",
                base_color=base_color,
            )

        kld_tf = float("nan")
        kld_auto = float("nan")
        if tf_emb is not None:
            kld_tf = float(state_space_kl(gt_emb, tf_emb, use_gmm=True))
            if save_figures:
                visualize_delay_embedding(
                    embedded=tf_emb,
                    save_dir=save_fig_dir,
                    variable_name=f"{safe_name}_tf_tau{time_delay}_d{delay_dims}",
                    explain=f"batch{batch_idx}_teacher_forced",
                    base_color="Greens",
                )
        if auto_emb is not None:
            kld_auto = float(state_space_kl(gt_emb, auto_emb, use_gmm=True))
            if save_figures:
                visualize_delay_embedding(
                    embedded=auto_emb,
                    save_dir=save_fig_dir,
                    variable_name=f"{safe_name}_auto_tau{time_delay}_d{delay_dims}",
                    explain=f"batch{batch_idx}_autonomous",
                    base_color="Reds",
                )

        per_channel[key] = {"kld_tf": kld_tf, "kld_auto": kld_auto}
        print(f"  {key} KLD TF: {kld_tf:.4f}  Auto: {kld_auto:.4f}")

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
        visualize_errors_from_lst(
            tf_scores,
            name_lst=tf_names,
            save_dir=save_fig_dir,
            explain=f"kld_tf_per_channel_batch{batch_idx}",
            error_unit="KLD",
            colors=tf_colors,
        )
        visualize_errors_from_lst(
            auto_scores,
            name_lst=auto_names,
            save_dir=save_fig_dir,
            explain=f"kld_auto_per_channel_batch{batch_idx}",
            error_unit="KLD",
            colors=auto_colors,
        )

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