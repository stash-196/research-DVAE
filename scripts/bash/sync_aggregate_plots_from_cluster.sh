#!/bin/bash

set -euo pipefail
#
# Sync only aggregated plot files from the cluster to a local desktop folder.
#
# Use case:
#   1. Edit the aggregated_results array below to list the remote plot folders.
#   2. Run a preview first:
#        bash scripts/bash/sync_aggregate_plots_from_cluster.sh --dry-run
#   3. If the output looks correct, run the real copy:
#        bash scripts/bash/sync_aggregate_plots_from_cluster.sh
#
# Edit the SSH target and the list of aggregated result directories below.

DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run|-n)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run|-n]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--dry-run|-n]" >&2
            exit 1
            ;;
    esac
done

# SSH config host alias, e.g. the one you use with `ssh deigo-ext`.
REMOTE_SSH_TARGET="deigo"

# Remote root used to validate incoming paths.
REMOTE_BASE_DIR="/bucket/DoyaU/stash/research-DVAE/saved_model"

# Local destination root on the desktop.
# LOCAL_BASE_DIR="$HOME/Downloads/phd-plots/plots/aggregate_plots"
LOCAL_BASE_DIR="$HOME/workspace/latex_projects/phd-thesis/Images/plots/aggregate_plots"

# Add the aggregated plot directories you want to copy.
aggregated_results=(
    # # 2026-05-20 
    # ## Lorenz
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-20/deigo_cluster/20260520-Lorenz_auto0-0.8_miss0-0.7_clip1_LossNone_LSTM_hdim20/aggregate_eval_plots_sampling_ratio_mask_label"
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-20/deigo_cluster/20260520-Lorenz_auto0-0.8_miss0-0.7_clip1_LossNone_MTRNN_hdim40/aggregate_eval_plots_sampling_ratio_mask_label"
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-20/deigo_cluster/20260520-Lorenz_auto0-0.8_miss0-0.7_clip1_LossNone_RNN_hdim40/aggregate_eval_plots_sampling_ratio_mask_label"
    # ## XHRO
    # ### MTRNN 
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-20/deigo_cluster/20260520-XHRO_ptf0.5-7_MTRNN_Subj70_ch3-4_hdim20-40_alphas3-9d/aggregate_eval_plots_sampling_ratio_observation_process_alphas=0_1, 0_1, 0_1_dim_rnn=40"
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-20/deigo_cluster/20260520-XHRO_ptf0.5-7_MTRNN_Subj70_ch3-4_hdim20-40_alphas3-9d/aggregate_eval_plots_sampling_ratio_observation_process_alphas=0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1_dim_rnn=40"
    # ### RNN
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-20/deigo_cluster/20260520-XHRO_ptf0.5-7_RNN_Subj70_ch3-4_hdim20-40/aggregate_eval_plots_sampling_ratio_observation_process_dim_rnn=40"

    # # 2026-05-21
    # ## MTRNN
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-21/deigo_cluster/20260521-XHRO_ptf0.5-7_MTRNN_Subj70_ch1-2_hdim20-40/aggregate_eval_plots_sampling_ratio_observation_process_alphas=0_1, 0_1, 0_1_dim_rnn=40"
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-21/deigo_cluster/20260521-XHRO_ptf0.5-7_MTRNN_Subj70_ch1-2_hdim20-40/aggregate_eval_plots_sampling_ratio_observation_process_alphas=0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1_dim_rnn=40"
    # ## RNN
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-21/deigo_cluster/20260521-XHRO_ptf0.5-7_RNN_Subj70_ch1-2_hdim20-40/aggregate_eval_plots_sampling_ratio_observation_process_dim_rnn=40"

    # # 2026-05-23
    # ## MTRNN 
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-23/deigo_cluster/20260523-XHRO_ptf0.5-7_MTRNN_Subj70_ch1-4_hdim200_alphas/aggregate_eval_plots_sampling_ratio_observation_process_alphas=0_1, 0_1, 0_1"
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-23/deigo_cluster/20260523-XHRO_ptf0.5-7_MTRNN_Subj70_ch1-4_hdim200_alphas/aggregate_eval_plots_sampling_ratio_observation_process_alphas=0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1, 0_1"
    # ## RNN
    # "/bucket/DoyaU/stash/research-DVAE/saved_model/2026-05-23/deigo_cluster/20260523-XHRO_ptf0.5-7_RNN_Subj70_ch1-4_hdim200/aggregate_eval_plots_sampling_ratio_observation_process"

    

)



# Copy only plot/image artifacts.
INCLUDE_PATTERNS=(
    "*.png"
    "*.pdf"
    "*.svg"
    "*.jpg"
    "*.jpeg"
)

mkdir -p "$LOCAL_BASE_DIR"

for remote_dir in "${aggregated_results[@]}"; do
    if [[ "$remote_dir" != "$REMOTE_BASE_DIR"/* ]]; then
        echo "[sync] Skipping path outside REMOTE_BASE_DIR: $remote_dir" >&2
        continue
    fi

    relative_dir="${remote_dir#*/deigo_cluster/}"
    if [[ "$relative_dir" == "$remote_dir" ]]; then
        echo "[sync] Skipping path without /deigo_cluster/: $remote_dir" >&2
        continue
    fi

    local_dir="$LOCAL_BASE_DIR/$relative_dir"
    mkdir -p "$local_dir"

    echo "[sync] Copying plots from: $remote_dir"
    echo "[sync] Local destination: $local_dir"

    rsync_args=(
        -av
        --prune-empty-dirs
    )

    if [[ "$DRY_RUN" == true ]]; then
        rsync_args+=(-n)
        echo "[sync] Dry run enabled; no files will be copied."
    fi

    for pattern in "${INCLUDE_PATTERNS[@]}"; do
        rsync_args+=(--include="*/" --include="$pattern")
    done
    rsync_args+=(--exclude="*")

    if [[ "$DRY_RUN" == true ]]; then
        printf '[sync] rsync command: rsync '
        printf '%q ' "${rsync_args[@]}" "${REMOTE_SSH_TARGET}:'${remote_dir%/}/'" "$local_dir/"
        printf '\n'
    fi

    rsync "${rsync_args[@]}" "${REMOTE_SSH_TARGET}:'${remote_dir%/}/'" "$local_dir/"
done

echo "[sync] Done."