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

ml singularity

# Run the Apptainer container
singularity run \
  --bind $PROJECT_PATH:/workspace/project \
  --bind $VENV_PATH:/workspace/venv \
  $CONTAINER_PATH \
  bin/train_model.py  --job_id $SLURM_JOBID --cfg /home/s/sutashu-tomonaga1/workspace/research-DVAE/config/general_signal/cfg_mt_rnn.ini
# Print the time again
echo "Time END: `date`"

