#!/bin/bash

#SBATCH --job-name=dvae_training     # Job name
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=16G                    # Total memory limit
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --output=./logs/%j_training.log     # Standard output and error log
#SBATCH --error=./logs/%j_training.err      # Standard output and error log
#SBATCH --partition=compute              # Specify the GPU partition


# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"
echo "Under SLURM JobID: $SLURM_JOBID"


cd ~/workspace/research-DVAE

# Define paths
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/

# Validate paths
for PATH_VAR in "$CONTAINER_PATH" "$PROJECT_PATH" "$VENV_PATH"; do
    if [ ! -d "$PATH_VAR" ] && [ ! -f "$PATH_VAR" ]; then
        echo "Error: $PATH_VAR does not exist"
        exit 1
    fi
done

ml singularity

# Run the Apptainer container
singularity run \
  --bind $PROJECT_PATH:/workspace/project \
  --bind $VENV_PATH:/workspace/venv \
  $CONTAINER_PATH \
  bin/train_model.py  --job_id $SLURM_JOBID --cfg /home/s/sutashu-tomonaga1/workspace/research-DVAE/config/general_signal/cfg_mt_rnn.ini

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Print the time again
echo "Time END: `date`"

