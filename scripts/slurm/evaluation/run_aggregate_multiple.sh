#!/bin/bash

# Script to generate and submit SLURM jobs for aggregating evaluation results using aggregate_evaluation_results.py
# Edit the experiments dict below to add/remove target directories

# Define experiment/parameter pairs using an associative array (dictionary).
# The key is the experiment directory, and the value is the full command-line arguments after the script path.
# Example: ["/path/to/exp"]="--parameters sampling_ratio mask_label --filter dim_rnn=64"
declare -A experiments=(
    ["/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_LSTM_hdi20_ptientHigh"]="--parameters sampling_ratio mask_label"
    ["/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_MTRNN_hdi20_ptientHigh"]="--parameters sampling_ratio mask_label"
    ["/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_hdi20s_ptientHigh"]="--parameters sampling_ratio mask_label"
    ["/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_MTRNN_hdi20-40_ptientHigh"]="--parameters sampling_ratio mask_label"

    # Add more key-value pairs here as needed: ["/path/to/exp"]="--parameters param1 param2 --filter key=val"
)

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Define paths
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Loop over each experiment directory in the dictionary
for EXPERIMENT_DIR in "${!experiments[@]}"; do
    FULL_ARGS="${experiments[$EXPERIMENT_DIR]}"

    if [ -z "$FULL_ARGS" ]; then
        echo "[bash] Skipping entry with empty arguments for directory: $EXPERIMENT_DIR"
        continue
    fi

    if [ ! -d "$EXPERIMENT_DIR" ]; then
        echo "[bash] Skipping missing experiment directory: $EXPERIMENT_DIR"
        continue
    fi

    # Parse parameters and filters from FULL_ARGS
    PARAMETERS=""
    FILTERS=""
    parsing_parameters=false
    parsing_filters=false
    for arg in $FULL_ARGS; do
        if [ "$arg" = "--parameters" ]; then
            parsing_parameters=true
        elif [ "$arg" = "--filter" ]; then
            parsing_filters=true
            FILTERS="$FILTERS --filter"
        elif [ "$parsing_filters" = true ]; then
            FILTERS="$FILTERS $arg"
            parsing_filters=false
        elif [ "$parsing_parameters" = true ]; then
            PARAMETERS="$PARAMETERS $arg"
        fi
    done

    # Trim leading spaces
    PARAMETERS=$(echo "$PARAMETERS" | sed 's/^ *//')
    FILTERS=$(echo "$FILTERS" | sed 's/^ *//')

    # Log directory
    LOG_DIR="$EXPERIMENT_DIR/aggregate_logs"
    echo "[bash] Processing experiment: $EXPERIMENT_DIR"
    echo "[bash] Parameters: $PARAMETERS"
    echo "[bash] Filters: $FILTERS"
    echo "[bash] LOG_DIR: $LOG_DIR"
    mkdir -p "$LOG_DIR"

    # Extract base name for the job
    EXPERIMENT_BASENAME=$(basename "$EXPERIMENT_DIR")

    # Compute container paths
    EXPERIMENT_CONTAINER_PATH=${EXPERIMENT_DIR/#$SAVED_HOST_PATH/\/saved_model}

    # Create output directory for plots inside the experiment dir
    OUTPUT_DIR_HOST="${EXPERIMENT_DIR}/aggregated_plots"
    OUTPUT_DIR_CONTAINER="${EXPERIMENT_CONTAINER_PATH}/aggregated_plots"
    mkdir -p "$OUTPUT_DIR_HOST"

    # Temporary SLURM script
    SLURM_SCRIPT="scripts/slurm/temp/run_agg_${EXPERIMENT_BASENAME}.slurm"

    cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH --job-name=${EXPERIMENT_BASENAME}_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/%j_agg_${EXPERIMENT_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_agg_${EXPERIMENT_BASENAME}.err
#SBATCH --partition=compute

# Define variables
CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
EXPERIMENT_DIR=$EXPERIMENT_DIR
LOG_DIR=$LOG_DIR

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] Log file: \${LOG_DIR}/%j_agg_${EXPERIMENT_BASENAME}.log"
echo "[slurm] EXPERIMENT_CONTAINER_PATH: $EXPERIMENT_CONTAINER_PATH"
echo "[slurm] Parameters: $PARAMETERS"
echo "[slurm] Filters: $FILTERS"

ml singularity

# Run the Apptainer container
singularity exec \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
    bash -c "source /workspace/venv/bin/activate && python3 src/dvae/eval/aggregate_evaluation_results.py $EXPERIMENT_CONTAINER_PATH --parameters $PARAMETERS $FILTERS --output_dir $OUTPUT_DIR_CONTAINER"

# Check exit code
EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code \$EXIT_CODE"
    exit \$EXIT_CODE
fi

echo "[slurm] Time END: \$(date)"
EOL

    # Submit the temporary SLURM script to the queue
    echo "[bash] Submitting aggregation for $EXPERIMENT_BASENAME"
    sbatch "$SLURM_SCRIPT"

done
