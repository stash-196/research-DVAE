#!/bin/bash

#SBATCH --job-name=vrnn_training     # Job name
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=16G                    # Total memory limit
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --output=./logs/collect_%j.log     # Standard output and error log
#SBATCH --error=./logs/collect_%j.err      # Standard output and error log
#SBATCH --partition=compute              # Specify the GPU partition


# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"
echo "Under SLURM JobID: $SLURM_JOBID"

# Activate environment, if there is one (e.g., conda or virtualenv)
source ~/miniconda3/bin/activate
# source activate environment_name
conda activate research-DVAE

cd ~/workspace/research-DVAE
# Run the command
~/miniconda3/envs/research-DVAE/bin/python collect_results.py

# Print the time again
echo "Time END: `date`"

