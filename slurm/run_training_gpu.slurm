#!/bin/bash

#SBATCH --job-name=gpu_dvae_training # Job name
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=1            # Number of CPU cores per task
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=./logs/gpu_training_%j.log     # Standard output and error log
#SBATCH --error=./logs/training_%j.err      # Standard output and error log
#SBATCH --partition=gpu              # Specify the GPU partition
#SBATCH --gres=gpu:1                 # Number of GPUs

# Print the time, hostname, and job ID
echo "Time: `date`"
echo "Running on host: `hostname`"sru
echo "Under SLURM JobID: $SLURM_JOBID"

# Activate  environment, if there is one (e.g., conda or virtualenv)
~/miniconda3/bin/activate
# source activate environment_name
conda activate research-DVAE

cd ~/workspace/research-DVAE
# Run the command
~/miniconda3/envs/research-DVAE/bin/python train_model.py --cfg ./config/lorenz63/cfg_vrnn.ini
