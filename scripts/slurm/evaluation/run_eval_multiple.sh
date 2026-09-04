#!/bin/bash

# Script to generate and submit SLURM jobs for evaluating multiple models using eval_signal.py
# Edit the experiments array below to add/remove target directories

# Define a list of experiment directories
declare -a experiments=(

    # "/flash/DoyaU/stash/research-DVAE/saved_model/2024-11-02/ep20000_8alphas_esp50_nanBers_ptf_MT-RNN_SampRatios"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2024-11-01/ep20000_8alphas_esp50_nanBers_ptf_MT-RNN_SampRatios"

#     2025-11-12/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-12/deigo_cluster/20251112_Lorenz_MissingHigh_ssptf_MTRNN-markovMiss_varySampRatios"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-12/deigo_cluster/20251112_Lorenz_MissingMedium_ssptf_MTRNN-markovMiss_varySampRatios"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-12/deigo_cluster/20251112_Lorenz_MissingNone_ssptf_MTRNN-markovMiss_varySampRatios"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-12/deigo_cluster/20251112_Lorenz_markovMissingHigh_ptf_MTRNN_varySampRatios_3alphas"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-12/deigo_cluster/20251112_Lorenz_markovMissingMedium_ptf_MTRNN_varySampRatios_3alphas"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-12/deigo_cluster/20251112_Lorenz_markovMissingNone_ptf_MTRNN_varySampRatios_3alphas"

# 2025-11-13/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-13/deigo_cluster/20251113_Lorenz_markovMissing0.8_ptf_MTRNN_varySampRatios_9alphas"

# 2025-11-14/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-14/deigo_cluster/20251114_Lorenz_markovMissing0.8_SS_MTRNN_varySampRatios_3or9alphas_allLoss"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-14/deigo_cluster/20251114_Lorenz_markovMissing0.8_ptf_MTRNN_varySampRatios_3or9alphas_allLoss"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-14/deigo_cluster/20251114_XHRO_ssHIGH-AllLoss_MTRNN_SampRatios_3Subjs_hdim256_alphaDim9_1Dchannel"

# # 2025-11-15/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-15/deigo_cluster/20251114_Lorenz_markovMissing0.8_SS_MTRNN_varySampRatios_3or9alphas_allLoss"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-15/deigo_cluster/20251114_Lorenz_markovMissing0.8_ptf_MTRNN_varySampRatios_3or9alphas_allLoss"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-15/deigo_cluster/20251114_XHRO_ssHIGH-AllLoss_MTRNN_SampRatios_3Subjs_hdim256_alphaDim9_1Dchannel"

# # 2026-01-14/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch"

# # 2026-01-16/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-16/deigo_cluster/20260116_XHRO_len500_drop0.1_ss0.1-AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-16/deigo_cluster/20260116_XHRO_ss0.1-AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-16/deigo_cluster/20260116_XHRO_sss-AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"

# # 2026-01-18/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-18/deigo_cluster/20260118_XHRO_len500_drop0.1_ss0.4-_AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"

# # 2026-01-21/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-21/deigo_cluster/20260121_XHRO_len1000_drop0_ss0.5-_AllLoss_MT-RNN_Subj70_ch3-4_h1000"

# # 2026-01-22/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-22/deigo_cluster/20260122_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_MT-RNN_Subj70_ch1-2_h1000"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-22/deigo_cluster/20260122_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_v-LSRNN_Subj70_ch3-4_h1000"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-22/deigo_cluster/20260122_XHRO_len1000_drop0_ptf0.6-_clip1_AllLoss_MT-RNN_Subj70_ch3-4_h1000"

# # 2026-01-24/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-24/deigo_cluster/20260123_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSRNN_Subj70_ch3-4_h100"

# # 2026-01-25/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-25/deigo_cluster/20260125_32mem_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-25/deigo_cluster/20260125_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100"

# # 2026-01-27/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-27/deigo_cluster/20260126_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-27/deigo_cluster/20260127_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_hdims"

# # 2026-01-28/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-28/deigo_cluster/20260128_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_hdims_ptientHigh"

# # 2026-01-29/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-29/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

# # 2026-01-30/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-30/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

# # 2026-02-06/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-06/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

# # 2026-02-12/
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_LSTM_hdi20_ptientHigh"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_MTRNN_hdi20_ptientHigh"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_hdi20s_ptientHigh"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_MTRNN_hdi20-40_ptientHigh"

# 2026-05-28/
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-05-28/deigo_cluster/20260528-Lorenz_auto0-0.8_miss0-0.7_clip1_ep20000_LossNone_MTRNN3-9d_hdim80"

# 2026-08-16/ resume_XhroPacketLoss realtime MTRNN-9d ptf 0.0-0.7
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-08-16/deigo_cluster/20260816-XHRO_packet_loss_ep20000_ptf0-7_MTRNN9d_clip10_chAll_4d_hdim200_eStop500"

    # Add more directories here as needed
)

# Optional: pass experiment dirs as args (overrides the list above)
if [ "$#" -gt 0 ]; then
    experiments=("$@")
fi

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Define paths (aligned with run_training.slurm)
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=/bucket/DoyaU/stash/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Ensure temp dir exists (relative to repo root when script is run from there)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/../temp"
mkdir -p "$TEMP_DIR"

# Loop over each experiment directory
for EXPERIMENT_DIR in "${experiments[@]}"; do
    # Create log directory under the experiment directory
    LOG_DIR="$EXPERIMENT_DIR/eval_logs"
    echo "[bash] Processing experiment: $EXPERIMENT_DIR"
    echo "[bash] LOG_DIR: $LOG_DIR"
    mkdir -p "$LOG_DIR"

    if [ ! -d "$EXPERIMENT_DIR" ]; then
        echo "[bash] Experiment directory does not exist: $EXPERIMENT_DIR"
        continue
    fi

# Prefer *final*.pt per run dir; else *checkpoint*.pt; skip if neither
find "$EXPERIMENT_DIR" -mindepth 1 -maxdepth 1 -type d | while read RUN_DIR; do
    RUN_BASENAME=$(basename "$RUN_DIR")
    case "$RUN_BASENAME" in
        logs|resume_logs|eval_logs|temp) continue ;;
    esac

    FINAL_FILE=$(find "$RUN_DIR" -maxdepth 1 -type f -name "*final*.pt" | head -n 1)
    CHECKPOINT_FILE=$(find "$RUN_DIR" -maxdepth 1 -type f -name "*checkpoint*.pt" | head -n 1)

    if [ -n "$FINAL_FILE" ]; then
        MODEL_FILE="$FINAL_FILE"
        WEIGHTS_SOURCE=final
    elif [ -n "$CHECKPOINT_FILE" ]; then
        MODEL_FILE="$CHECKPOINT_FILE"
        WEIGHTS_SOURCE=checkpoint
    else
        echo "[bash] SKIP (no final/checkpoint weights): $RUN_DIR"
        continue
    fi

    echo "=============================================="
    echo "[bash] WEIGHTS_SOURCE=$WEIGHTS_SOURCE"
    echo "[bash] RUN_DIR=$RUN_DIR"
    echo "[bash] MODEL_FILE=$MODEL_FILE"
    echo "=============================================="

    # Extract the base name of the model file for the job name
    MODEL_BASENAME=$(basename "$MODEL_FILE" .pt)
    # Unique short tag so concurrent ptf jobs do not clobber the same temp slurm file
    PTF_TAG=$(echo "$RUN_BASENAME" | grep -oE 'ptf_[0-9.]+' | head -n 1)
    if [ -z "$PTF_TAG" ]; then
        PTF_TAG=$(echo "$RUN_BASENAME" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-48)
    fi
    UNIQUE_NAME="${PTF_TAG}_${MODEL_BASENAME}"

    # Compute the container-internal path for the model file
    MODEL_CONTAINER_PATH=${MODEL_FILE/#$SAVED_HOST_PATH/\/saved_model}

    # Create a temporary SLURM script for this model
    cat > "$TEMP_DIR/run_eval_$UNIQUE_NAME.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=${UNIQUE_NAME}_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=${LOG_DIR}/%j_eval_${UNIQUE_NAME}.log
#SBATCH --error=${LOG_DIR}/%j_eval_${UNIQUE_NAME}.err
#SBATCH --partition=compute

# Define variables
CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
EXPERIMENT_DIR=$EXPERIMENT_DIR
MODEL_FILE=$MODEL_FILE
LOG_DIR=$LOG_DIR
MODEL_BASENAME=$MODEL_BASENAME
UNIQUE_NAME=$UNIQUE_NAME
WEIGHTS_SOURCE=$WEIGHTS_SOURCE
MODEL_CONTAINER_PATH=$MODEL_CONTAINER_PATH

# Print the time, hostname, and job ID
echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] WEIGHTS_SOURCE=\$WEIGHTS_SOURCE"
echo "[slurm] Log file: \${LOG_DIR}/%j_eval_\${UNIQUE_NAME}.log"
echo "[slurm] MODEL_CONTAINER_PATH: \$MODEL_CONTAINER_PATH"

# Check if model file exists on host
if [ ! -f "\$MODEL_FILE" ]; then
    echo "[slurm] Error: Model file \$MODEL_FILE does not exist on host"
    exit 1
fi

# Validate paths
for PATH_VAR in "\$CONTAINER_PATH" "\$PROJECT_PATH" "\$VENV_PATH" "\$DATA_HOST_PATH" "\$SAVED_HOST_PATH"; do
    if [ ! -d "\$PATH_VAR" ] && [ ! -f "\$PATH_VAR" ]; then
        echo "Error: \$PATH_VAR does not exist"
        exit 1
    fi
done

# Ensure Lmod/ml is available on compute nodes (non-login bash)
if ! type ml >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile 2>/dev/null || true
fi
# Initialize Modules (compute nodes expose singularity under /apps MODULEPATH)
if [ -f /etc/profile.d/zz_deigo_base.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/zz_deigo_base.sh
fi
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
ml singularity

# Set environment variables to prevent buffering and thread oversubscription
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
NO_VIZ_FLAG=""
if [ "${NO_VIZ:-0}" = "1" ]; then
  NO_VIZ_FLAG="--no-viz"
  echo "[slurm] NO_VIZ=1 -> passing --no-viz"
fi

# Run the Apptainer container (cwd = project root on host, bound as /workspace/project)
singularity exec \\
  --pwd /workspace/project \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
  bash -c "source /workspace/venv/bin/activate && python3 src/dvae/eval/eval_signal.py --saved_dict \$MODEL_CONTAINER_PATH --weights-source \$WEIGHTS_SOURCE --save-3d False \$NO_VIZ_FLAG"

# Check exit code
EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code \$EXIT_CODE"
    exit \$EXIT_CODE
fi

# Print the time again
echo "[slurm] Time END: \$(date)"
EOL

    # Submit the temporary SLURM script to the queue
    echo "[bash] Submitting eval for $UNIQUE_NAME (WEIGHTS_SOURCE=$WEIGHTS_SOURCE)"
    sbatch "$TEMP_DIR/run_eval_$UNIQUE_NAME.slurm"

    # Optionally, remove the temporary SLURM script after submission
    # rm "$TEMP_DIR/run_eval_$UNIQUE_NAME.slurm"
done
done