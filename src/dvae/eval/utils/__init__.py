"""Eval utilities.

Submodules are imported explicitly. Names re-exported here load on first use
so a caller can import a light helper (delay embedding, intrinsic dimension)
without pulling sklearn / the visualizer stack.
"""

from importlib import import_module

_EXPORTS = {
    "run_spectrum_analysis": (
        "dvae.eval.utils.frequency_analysis",
        "run_spectrum_analysis",
    ),
    "compute_delay_embedding": (
        "dvae.eval.utils.delay_embedding",
        "compute_delay_embedding",
    ),
    "power_spectrum_error": (
        "dvae.eval.utils.durstewitz_eval_metrics",
        "power_spectrum_error",
    ),
    "normalize_and_smooth_power_spectrum": (
        "dvae.eval.utils.durstewitz_eval_metrics",
        "normalize_and_smooth_power_spectrum",
    ),
    "hellinger_distance": (
        "dvae.eval.utils.durstewitz_eval_metrics",
        "hellinger_distance",
    ),
    "n_step_prediction_error": (
        "dvae.eval.utils.durstewitz_eval_metrics",
        "n_step_prediction_error",
    ),
    "state_space_kl": ("dvae.eval.utils.durstewitz_eval_metrics", "state_space_kl"),
    "run_mse_analysis": ("dvae.eval.utils.run_mse_analysis", "run_mse_analysis"),
    "run_geometry_analysis": (
        "dvae.eval.utils.run_geometry_analysis",
        "run_geometry_analysis",
    ),
    "compute_local_drift_statistics": (
        "dvae.eval.utils.local_drift_analysis",
        "compute_local_drift_statistics",
    ),
    "run_forward_with_mode": (
        "dvae.eval.utils.forward_modes",
        "run_forward_with_mode",
    ),
    "build_mode_selector": ("dvae.eval.utils.forward_modes", "build_mode_selector"),
    "get_flip_point_for_mode": (
        "dvae.eval.utils.forward_modes",
        "get_flip_point_for_mode",
    ),
    "get_auto_mask_1d": ("dvae.eval.utils.forward_modes", "get_auto_mask_1d"),
    "mode_selector_to_1d": ("dvae.eval.utils.forward_modes", "mode_selector_to_1d"),
    "count_auto_blocks": ("dvae.eval.utils.forward_modes", "count_auto_blocks"),
    "list_auto_block_ranges": (
        "dvae.eval.utils.forward_modes",
        "list_auto_block_ranges",
    ),
    "get_channel_benchmarks": (
        "dvae.eval.utils.benchmark_signals",
        "get_channel_benchmarks",
    ),
    "merge_batch_metric_dicts": (
        "dvae.eval.utils.metrics_aggregate",
        "merge_batch_metric_dicts",
    ),
    "flatten_analysis_to_batch_metrics": (
        "dvae.eval.utils.metrics_aggregate",
        "flatten_analysis_to_batch_metrics",
    ),
    "collect_batch_visual_record": (
        "dvae.eval.utils.batch_all_visuals",
        "collect_batch_visual_record",
    ),
    "compute_stitched_kld_metrics": (
        "dvae.eval.utils.batch_all_visuals",
        "compute_stitched_kld_metrics",
    ),
    "render_batch_all_visuals": (
        "dvae.eval.utils.batch_all_visuals",
        "render_batch_all_visuals",
    ),
    "render_summary_error_bars": (
        "dvae.eval.utils.batch_all_visuals",
        "render_summary_error_bars",
    ),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(list(globals().keys()) + list(_EXPORTS.keys())))
