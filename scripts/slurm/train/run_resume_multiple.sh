#!/bin/bash

# Script to generate and submit SLURM jobs for resuming training of models
# Set DRY_RUN=1 to test without submitting (prints sbatch commands instead)
# Set DRY_RUN=0 to actually submit jobs
DRY_RUN=1

# Edit the experiments array below to add/remove target directories

# Define a list of experiment directories
declare -a experiments=(

    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_LSTM_hdi20_ptientHigh"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_MTRNN_hdi20_ptientHigh"

   "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-30/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh"

    # Add more directories here as needed
)

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Define paths (aligned with run_training_multiple.sh)
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Compute script directory and temp dir relative to this script (robust to caller cwd)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/../temp"
mkdir -p "$TEMP_DIR"

# Loop over each experiment directory
for EXPERIMENT_DIR in "${experiments[@]}"; do
    # Create log directory under the experiment directory
    LOG_DIR="$EXPERIMENT_DIR/resume_logs"
    echo "[bash] Processing experiment: $EXPERIMENT_DIR"

    if [ \! -d "$EXPERIMENT_DIR" ]; then
        echo "[bash] Experiment directory does not exist: $EXPERIMENT_DIR"
        continue
    fi

    echo "[bash] LOG_DIR: $LOG_DIR"
    mkdir -p "$LOG_DIR"

    # Find all subdirectories in the EXPERIMENT_DIR
    find "$EXPERIMENT_DIR" -mindepth 1 -maxdepth 1 -type d | while read RUN_DIR; do

        # Check if checkpoint exists
        CHECKPOINT_FILE=$(find "$RUN_DIR" -maxdepth 1 -name "*checkpoint.pt" | head -n 1)

        # Check if final file exists
        FINAL_FILE=$(find "$RUN_DIR" -maxdepth 1 -name "*final*.pt" | head -n 1)

        if [ -n "$CHECKPOINT_FILE" ] && [ -z "$FINAL_FILE" ]; then
            echo "[bash] Found incomplete run: $RUN_DIR"

            # The base name of the run directory
            RUN_BASENAME=$(basename "$RUN_DIR")

            # Container-internal path for model dir
            RUN_CONTAINER_PATH=${RUN_DIR/#$SAVED_HOST_PATH/\/saved_model}

            # We also need the config.ini file inside the directory
            if [ \! -f "$RUN_DIR/config.ini" ]; then
                echo "[bash] Warning: No config.ini found in $RUN_DIR, skipping..."
                continue
            fi

            CONFIG_CONTAINER_PATH="${RUN_CONTAINER_PATH}/config.ini"

# Create a temporary SLURM script for this run
            cat > "$TEMP_DIR/run_resume_$RUN_BASENAME.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=${RUN_BASENAME}_resume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=${LOG_DIR}/%j_resume_${RUN_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_resume_${RUN_BASENAME}.err
#SBATCH --partition=compute

# Define variables
CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
EXPERIMENT_DIR=$EXPERIMENT_DIR
RUN_DIR=$RUN_DIR
LOG_DIR=$LOG_DIR
RUN_BASENAME=$RUN_BASENAME
RUN_CONTAINER_PATH=$RUN_CONTAINER_PATH
CONFIG_CONTAINER_PATH=$CONFIG_CONTAINER_PATH

# Print the time, hostname, and job ID
echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] Log file: \${LOG_DIR}/%j_resume_\${RUN_BASENAME}.log"
echo "[slurm] Resuming model directory: \$RUN_CONTAINER_PATH"

# Validate paths
for PATH_VAR in "\$CONTAINER_PATH" "\$PROJECT_PATH" "\$VENV_PATH" "\$DATA_HOST_PATH" "\$SAVED_HOST_PATH"; do
    if [ \! -d "\$PATH_VAR" ] && [ \! -f "\$PATH_VAR" ]; then
        echo "Error: \$PATH_VAR does not exist"
        exit 1
    fi
done

ml singularity

# Run the Apptainer container
singularity run \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
  bin/train_model.py --job_id \$SLURM_JOBID --cfg \$CONFIG_CONTAINER_PATH --reload --model_dir \$RUN_CONTAINER_PATH

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
            echo "[bash] Submitting resume job for $RUN_BASENAME"
            if [ "$DRY_RUN" -eq 1 ]; then
                echo "[bash] [DRY RUN] sbatch $TEMP_DIR/run_resume_$RUN_BASENAME.slurm"
            else
                sbatch "$TEMP_DIR/run_resume_$RUN_BASENAME.slurm"
            fi

            # Optionally, remove the temporary SLURM script after submission
            # rm "$TEMP_DIR/run_resume_$RUN_BASENAME.slurm"
        fi
    done
done
