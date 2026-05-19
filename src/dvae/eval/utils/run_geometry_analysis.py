from dvae.eval.utils import (
    compute_delay_embedding,
    state_space_kl,
)
from dvae.visualizers import visualize_delay_embedding
from dvae.visualizers.visualizers import visualize_errors_from_lst
from dvae.eval.utils.benchmark_signals import get_benchmark_signals

# Mapping from color names to matplotlib colormap names for base_color
color_to_base = {
    "blue": "Blues",
    "green": "Greens",
    "red": "Reds",
    "orange": "Oranges",
    "magenta": "Purples",
    "cyan": "GnBu",  # Cyan maps to GnBu (green-blue)
    "yellow": "YlOrBr",
}


def run_geometry_analysis(
    test_dataloader,
    recon_data_long,
    save_fig_dir,
    i,
    autonomous_mode_selector_long,
    dataset_name,
    batch_data_long=None,
):
    sig_info = get_benchmark_signals(
        dataset_name,
        test_dataloader,
        i,
        recon_data_long,
        autonomous_mode_selector_long,
        batch_data_long,
    )
    long_data_lst = sig_info["long_data_lst"]
    name_lst = sig_info["name_lst"]
    key_lst = sig_info["key_lst"]
    true_signal_index = sig_info["true_signal_index"]
    colors_lst = sig_info["colors_lst"]

    time_delay = sig_info["time_delay"]
    delay_dims = sig_info["delay_dims"]

    embeddings_lst = []
    # Make sure we don't compute embeddings for sequences that are too short.
    # The minimum required length is roughly time_delay * (delay_dims - 1) + 1.
    for sig in long_data_lst:
        emb = compute_delay_embedding(
            observation=sig,
            delay=time_delay,
            dimensions=delay_dims,
        )
        embeddings_lst.append(emb)

    for name, key, emb, color in zip(name_lst, key_lst, embeddings_lst, colors_lst):
        # We replace newline since name might have 'Ground\nTruth'
        safe_name = name.replace("\n", "_").replace(" ", "_").lower()
        base_color = color_to_base.get(color, "Blues")  # Default to Blues if not found
        visualize_delay_embedding(
            embedded=emb,
            save_dir=save_fig_dir,
            variable_name=f"{safe_name}_tau{time_delay}_d{delay_dims}",
            explain="full_trajectory_delay",
            base_color=base_color,
        )

    gt_embedding = embeddings_lst[true_signal_index]
    print("[Eval] KLD (State-Space via Delay Embedding):")
    kld_scores = []

    for j, (name, emb) in enumerate(zip(name_lst, embeddings_lst)):
        if j == true_signal_index:
            kld = 0.0
        else:
            kld = state_space_kl(
                true_traj=gt_embedding,
                gen_traj=emb,
                use_gmm=True,
            )

        kld_scores.append(float(kld))
        print(f"  KLD {name.replace(chr(10), ' ')}: {kld:.4f}")

    # Visualize KLD error bars
    visualize_errors_from_lst(
        kld_scores,
        name_lst=name_lst,
        save_dir=save_fig_dir,
        explain="kld_error_per_signal",
        error_unit="KLD",
        colors=colors_lst,
    )

    return {
        "kld_scores": kld_scores,
        "signal_keys": key_lst,
    }
