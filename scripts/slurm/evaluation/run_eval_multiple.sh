#!/bin/bash

# Script to generate and submit SLURM jobs for evaluating multiple models using eval_signal.py
# Usage: ./run_eval_multiple.sh <experiment_directory>

if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment_directory>"
    exit 1
fi

EXPERIMENT_DIR=$1

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Define paths (aligned with run_training.slurm)
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Create log directory under the experiment directory
LOG_DIR="$EXPERIMENT_DIR/eval_logs"
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