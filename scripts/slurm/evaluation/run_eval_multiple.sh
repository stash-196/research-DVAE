#!/bin/bash

# Script to generate and submit SLURM jobs for evaluating multiple models using eval_signal.py
# Edit the experiments array below to add/remove target directories

# Define a list of experiment directories
declare -a experiments=(

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

# 2025-11-15/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-15/deigo_cluster/20251114_Lorenz_markovMissing0.8_SS_MTRNN_varySampRatios_3or9alphas_allLoss"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-15/deigo_cluster/20251114_Lorenz_markovMissing0.8_ptf_MTRNN_varySampRatios_3or9alphas_allLoss"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2025-11-15/deigo_cluster/20251114_XHRO_ssHIGH-AllLoss_MTRNN_SampRatios_3Subjs_hdim256_alphaDim9_1Dchannel"

# 2026-01-14/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch"

# 2026-01-16/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-16/deigo_cluster/20260116_XHRO_len500_drop0.1_ss0.1-AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-16/deigo_cluster/20260116_XHRO_ss0.1-AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-16/deigo_cluster/20260116_XHRO_sss-AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"

# 2026-01-18/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-18/deigo_cluster/20260118_XHRO_len500_drop0.1_ss0.4-_AllLoss_v-LS-sh-PL-RNN_Subj70_ch4_h1000"

# 2026-01-21/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-21/deigo_cluster/20260121_XHRO_len1000_drop0_ss0.5-_AllLoss_MT-RNN_Subj70_ch3-4_h1000"

# 2026-01-22/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-22/deigo_cluster/20260122_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_MT-RNN_Subj70_ch1-2_h1000"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-22/deigo_cluster/20260122_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_v-LSRNN_Subj70_ch3-4_h1000"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-22/deigo_cluster/20260122_XHRO_len1000_drop0_ptf0.6-_clip1_AllLoss_MT-RNN_Subj70_ch3-4_h1000"

# 2026-01-24/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-24/deigo_cluster/20260123_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSRNN_Subj70_ch3-4_h100"

# 2026-01-25/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-25/deigo_cluster/20260125_32mem_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-25/deigo_cluster/20260125_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100"

# 2026-01-27/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-27/deigo_cluster/20260126_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100"
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-27/deigo_cluster/20260127_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_hdims"

# 2026-01-28/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-28/deigo_cluster/20260128_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_hdims_ptientHigh"

# 2026-01-29/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-29/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

# 2026-01-30/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-30/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

# 2026-02-06/
    "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-06/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

# 2026-02-12/
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_LSTM_hdi20_ptientHigh"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_MTRNN_hdi20_ptientHigh"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_hdi20s_ptientHigh"
#     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_MTRNN_hdi20-40_ptientHigh"

    # Add more directories here as needed
)

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Define paths (aligned with run_training.slurm)
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Loop over each experiment directory
for EXPERIMENT_DIR in "${experiments[@]}"; do
    # Create log directory under the experiment directory
    LOG_DIR="$EXPERIMENT_DIR/eval_logs"
    echo "[bash] Processing experiment: $EXPERIMENT_DIR"
    echo "[bash] LOG_DIR: $LOG_DIR"
    mkdir -p "$LOG_DIR"

# Find all .pt files containing 'final' in subdirectories of EXPERIMENT_DIR
find "$EXPERIMENT_DIR" -type f -name "*final*.pt" | while read MODEL_FILE; do
    # Extract the base name of the model file for the job name
    MODEL_BASENAME=$(basename "$MODEL_FILE" .pt)

    # Compute the container-internal path for the model file
    MODEL_CONTAINER_PATH=${MODEL_FILE/#$SAVED_HOST_PATH/\/saved_model}

    # Create a temporary SLURM script for this model
    cat > "scripts/slurm/temp/run_eval_$MODEL_BASENAME.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=${MODEL_BASENAME}_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=${LOG_DIR}/%j_eval_${MODEL_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_eval_${MODEL_BASENAME}.err
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
MODEL_CONTAINER_PATH=$MODEL_CONTAINER_PATH

# Print the time, hostname, and job ID
echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] Log file: \${LOG_DIR}/%j_eval_\${MODEL_BASENAME}.log"
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

ml singularity

# Run the Apptainer container
singularity exec \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
  bash -c "source /workspace/venv/bin/activate && python3 src/dvae/eval/eval_signal.py --saved_dict \$MODEL_CONTAINER_PATH"

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
    echo "[bash] Submitting eval for $MODEL_BASENAME"
    sbatch "scripts/slurm/temp/run_eval_$MODEL_BASENAME.slurm"

    # Optionally, remove the temporary SLURM script after submission
    # rm "scripts/slurm/temp/run_eval_$MODEL_BASENAME.slurm"
done
done