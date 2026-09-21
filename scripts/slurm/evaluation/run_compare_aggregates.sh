#!/bin/bash

# Overlay finished per-sweep aggregates (sibling of run_aggregate_multiple.sh).
#
# Usage (from the repo root on Deigo, after a git pull):
#   bash scripts/slurm/evaluation/run_compare_aggregates.sh
#
# Queue behind aggregate job IDs without babysitting (comma or colon ok):
#   DEPENDENCY_JOBIDS=123,456 bash scripts/slurm/evaluation/run_compare_aggregates.sh
#   bash scripts/slurm/evaluation/run_compare_aggregates.sh --dependency 123:456
#
# That becomes: sbatch --dependency=afterok:123:456 ...
#
# Edit EXPERIMENTS below (comment-toggle). Each entry is "label|/host/path".
# Path may be the experiment saved_model dir or its aggregate_eval_plots_* dir.
# The Python job resolves aggregated_values.csv at runtime (so it is fine if
# the CSV does not exist yet when this launcher is submitted with afterok).
#
# Optional env overrides:
#   METRICS="kld_auto spectrum_error_auto kld_tf spectrum_error_tf"
#   X_PARAMETER=sampling_ratio
#   OUTPUT_DIR=/flash/.../saved_model/compare_aggregates/my_name
#   COMPARE_NAME=otf_interpolate_indicate
#   DEPENDENCY_JOBIDS=123,456
#   CHANNEL_PAIR=1          # 1d-sep vs joint-4d per-channel join
#
# Or pass --channel-pair on the launcher (same as CHANNEL_PAIR=1).
# After the 1d-sep full re-agg exists, compare it to Jul-1 4d OTF with:
#   CHANNEL_PAIR=1 COMPARE_NAME=1d_vs_4d_otf bash scripts/slurm/evaluation/run_compare_aggregates.sh
# after editing EXPERIMENTS to the 1d-sep aggregate dir + the Jul-1 4d OTF root.

# Comment-toggle the comparison set. Labels appear in the plot legend.
EXPERIMENTS=(
    # 2026-07-01/ original 4d XHRO OTF (re-eval + aggregate for fair Sep-4 compare)
    # "OTF|/flash/DoyaU/stash/research-DVAE/saved_model/2026-07-01/deigo_cluster/20260701-XHRO_ep20000_ptf0-8_MTRNN9d_clip10_Subj70_chAll_4d_hdim200_eStop500"

    # 2026-09-04/ Jul-1-style 4d interpolate + indicate
    # "interpolate|/flash/DoyaU/stash/research-DVAE/saved_model/2026-09-04/deigo_cluster/20260904-XHRO_ep20000_ptf0-7_MTRNN9d_clip10_Subj70_chAll_4d_hdim200_eStop500_interpolate"
    # "indicate|/flash/DoyaU/stash/research-DVAE/saved_model/2026-09-04/deigo_cluster/20260904-XHRO_ep20000_ptf0-7_MTRNN9d_clip10_Subj70_chAll_4d_indicate_x8_hdim200_eStop500"

    # Example: point at an aggregate dir instead of the experiment root
    # "OTF|/flash/DoyaU/stash/research-DVAE/saved_model/2026-07-01/deigo_cluster/20260701-XHRO_ep20000_ptf0-8_MTRNN9d_clip10_Subj70_chAll_4d_hdim200_eStop500/aggregate_eval_plots_sampling_ratio"

    # 1d-sep vs Jul-1 4d OTF (needs CHANNEL_PAIR=1 and a finished 1d re-agg)
    "1d-sep|/flash/DoyaU/stash/research-DVAE/saved_model/2026-09-11/deigo_cluster/20260911-XHRO_ep20000_ptf0-7_MTRNN9d_clip10_Subj70_raw_ch1-4_1d_sep_hdim200_eStop500"
    "OTF-4d|/flash/DoyaU/stash/research-DVAE/saved_model/2026-07-01/deigo_cluster/20260701-XHRO_ep20000_ptf0-8_MTRNN9d_clip10_Subj70_chAll_4d_hdim200_eStop500"
)

METRICS="${METRICS:-kld_auto spectrum_error_auto kld_tf spectrum_error_tf}"
X_PARAMETER="${X_PARAMETER:-sampling_ratio}"
COMPARE_NAME="${COMPARE_NAME:-otf_interpolate_indicate}"

DEPENDENCY_JOBIDS="${DEPENDENCY_JOBIDS:-}"
CHANNEL_PAIR="${CHANNEL_PAIR:-0}"
while [ $# -gt 0 ]; do
    case "$1" in
        --dependency)
            DEPENDENCY_JOBIDS="$2"
            shift 2
            ;;
        --dependency=*)
            DEPENDENCY_JOBIDS="${1#*=}"
            shift
            ;;
        --channel-pair)
            CHANNEL_PAIR=1
            shift
            ;;
        --channel-pair=*)
            CHANNEL_PAIR="${1#*=}"
            shift
            ;;
        *)
            echo "[bash] Unknown argument: $1" >&2
            echo "[bash] Usage: $0 [--dependency JOBID[,JOBID...]] [--channel-pair]" >&2
            exit 1
            ;;
    esac
done

CHANNEL_PAIR_FLAG=""
case "${CHANNEL_PAIR}" in
    1|true|TRUE|yes|YES|on|ON)
        CHANNEL_PAIR_FLAG="--channel-pair"
        echo "[bash] Channel-pair mode: on"
        ;;
esac

# Paths (aligned with run_aggregate_multiple.sh)
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=/bucket/DoyaU/stash/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

today=$(date +%Y-%m-%d)
OUTPUT_DIR_HOST="${OUTPUT_DIR:-$SAVED_HOST_PATH/compare_aggregates/${today}_${COMPARE_NAME}}"
OUTPUT_DIR_CONTAINER="${OUTPUT_DIR_HOST/#$SAVED_HOST_PATH/\/saved_model}"
LOG_DIR="$OUTPUT_DIR_HOST/logs"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/../temp"
mkdir -p "$TEMP_DIR" "$OUTPUT_DIR_HOST" "$LOG_DIR"

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "[bash] EXPERIMENTS is empty. Uncomment label|path entries and retry." >&2
    exit 1
fi

# Build --experiments args with container paths; warn if the host path is missing
# (CSV may still appear later when DEPENDENCY_JOBIDS waits on aggregate jobs).
EXPERIMENT_ARGS=()
for entry in "${EXPERIMENTS[@]}"; do
    if [[ "$entry" == *"|"* ]]; then
        label="${entry%%|*}"
        host_path="${entry#*|}"
    elif [[ "$entry" == *"="* ]]; then
        label="${entry%%=*}"
        host_path="${entry#*=}"
    else
        echo "[bash] Invalid EXPERIMENTS entry (need label|path): $entry" >&2
        exit 1
    fi
    if [ ! -e "$host_path" ]; then
        echo "[bash] Warning: path does not exist yet (ok if waiting on afterok): $host_path"
    fi
    container_path="${host_path/#$SAVED_HOST_PATH/\/saved_model}"
    EXPERIMENT_ARGS+=("${label}|${container_path}")
    echo "[bash] ${label} -> ${host_path}"
done

SBATCH_DEP=""
if [ -n "$DEPENDENCY_JOBIDS" ]; then
    DEP_CLEAN="${DEPENDENCY_JOBIDS// /}"
    DEP_CLEAN="${DEP_CLEAN#afterok:}"
    DEP_CLEAN="${DEP_CLEAN//,/:}"
    if [ -z "$DEP_CLEAN" ]; then
        echo "[bash] DEPENDENCY_JOBIDS/ --dependency was set but empty after parsing." >&2
        exit 1
    fi
    SBATCH_DEP="--dependency=afterok:${DEP_CLEAN}"
    echo "[bash] Slurm dependency: $SBATCH_DEP"
fi

SLURM_SCRIPT="$TEMP_DIR/run_compare_aggregates.slurm"
EXP_ARGS_STR=""
for spec in "${EXPERIMENT_ARGS[@]}"; do
    EXP_ARGS_STR+=" $(printf '%q' "$spec")"
done

cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH --job-name=compare_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=${LOG_DIR}/%j_compare_aggregates.log
#SBATCH --error=${LOG_DIR}/%j_compare_aggregates.err
#SBATCH --partition=compute

CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
OUTPUT_DIR_CONTAINER=$OUTPUT_DIR_CONTAINER

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] Experiments:${EXP_ARGS_STR}"
echo "[slurm] Metrics: $METRICS"
echo "[slurm] X parameter: $X_PARAMETER"
echo "[slurm] Channel-pair: ${CHANNEL_PAIR_FLAG:-off}"
echo "[slurm] Output: $OUTPUT_DIR_CONTAINER"

if [ -f /etc/profile.d/zz_deigo_base.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/zz_deigo_base.sh
fi
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
ml singularity

singularity exec \\
  --pwd /workspace/project \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
    bash -c "source /workspace/venv/bin/activate && python3 src/dvae/eval/compare_aggregated_results.py --experiments${EXP_ARGS_STR} --metrics $METRICS --x-parameter $X_PARAMETER --output_dir \$OUTPUT_DIR_CONTAINER $CHANNEL_PAIR_FLAG"

EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code \$EXIT_CODE"
    exit \$EXIT_CODE
fi

echo "[slurm] Time END: \$(date)"
EOL

echo "[bash] Output dir: $OUTPUT_DIR_HOST"
echo "[bash] Metrics: $METRICS"
if [ -n "$CHANNEL_PAIR_FLAG" ]; then
    echo "[bash] Channel-pair: $CHANNEL_PAIR_FLAG"
fi
echo "[bash] Submitting compare_aggregates"
if [ -n "$SBATCH_DEP" ]; then
    sbatch $SBATCH_DEP "$SLURM_SCRIPT"
else
    sbatch "$SLURM_SCRIPT"
fi
