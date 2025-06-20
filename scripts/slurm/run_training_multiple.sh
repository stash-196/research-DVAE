#!/bin/bash

# Base directory where the configurations are stored
BASE_DIR=~/workspace/research-DVAE/config/general_signal/generated

# Define a list of experiment names, each corresponding to a subdirectory under BASE_DIR
declare -a experiments=("20250615-MT-VandVanillaRNN-optimize_alphas?-False")

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Get the current time in HH:MM:SS format
now=$(date +%H:%M:%S)

# Define paths (aligned with run_training.slurm)
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Create output directory for today using SAVED_HOST_PATH
OUTPUT_TODAY_DIR="$SAVED_HOST_PATH/$today"
echo "[bash] OUTPUT_TODAY_DIR: $OUTPUT_TODAY_DIR"
mkdir -p "$OUTPUT_TODAY_DIR"

# Loop over each experiment name to process its configuration files
for experiment in "${experiments[@]}"; do
    CONFIG_DIR="$BASE_DIR/$experiment"
    LOG_DIR="$OUTPUT_TODAY_DIR/$experiment/logs"
    echo "[bash] LOG_DIR: $LOG_DIR"
    mkdir -p "$LOG_DIR"

    # Use the find command to locate .ini files and iterate over them
    find "$CONFIG_DIR" -name "*.ini" | while read CONFIG_FILE; do
        # Extract the base name of the configuration file for the job name
        CONFIG_BASENAME=$(basename "$CONFIG_FILE" .ini)

        # Compute the container-internal path for the config file
        CONFIG_CONTAINER_PATH=${CONFIG_FILE/#$PROJECT_PATH/\/workspace\/project}

        # Create a temporary SLURM script for this configuration
        cat > "scripts/slurm/temp/run_training_$CONFIG_BASENAME.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=${CONFIG_BASENAME}_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=${LOG_DIR}/%j_training_${CONFIG_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_training_${CONFIG_BASENAME}.err
#SBATCH --partition=compute

# Define variables
CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
today=$today
experiment=$experiment
CONFIG_DIR=$CONFIG_DIR
CONFIG_FILE=$CONFIG_FILE
LOG_DIR=$LOG_DIR
CONFIG_BASENAME=$CONFIG_BASENAME
CONFIG_CONTAINER_PATH=$CONFIG_CONTAINER_PATH

# Print the time, hostname, and job ID
echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] Log file: \${LOG_DIR}/%j_training_\${CONFIG_BASENAME}.log"

# Create directories under SAVED_HOST_PATH and copy params_being_compared.txt
OUTPUT_EXPERIMENT_DIR="\${SAVED_HOST_PATH}/\${today}/\${experiment}"
echo "[slurm] OUTPUT_EXPERIMENT_DIR: \$OUTPUT_EXPERIMENT_DIR"
mkdir -p "\$OUTPUT_EXPERIMENT_DIR"
cp "\${CONFIG_DIR}/params_being_compared.txt" "\$OUTPUT_EXPERIMENT_DIR/"

# Validate paths
for PATH_VAR in "\$CONTAINER_PATH" "\$PROJECT_PATH" "\$VENV_PATH" "\$DATA_HOST_PATH" "\$SAVED_HOST_PATH"; do
    if [ ! -d "\$PATH_VAR" ] && [ ! -f "\$PATH_VAR" ]; then
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
  bin/train_model.py --job_id \$SLURM_JOBID --cfg \$CONFIG_CONTAINER_PATH

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
        echo "[bash] Submitting $experiment / run_training_$CONFIG_BASENAME.slurm"
        sbatch "scripts/slurm/temp/run_training_$CONFIG_BASENAME.slurm"

        # Optionally, remove the temporary SLURM script after submission
        # rm "scripts/slurm/temp/run_training_$CONFIG_BASENAME.slurm"
    done
done