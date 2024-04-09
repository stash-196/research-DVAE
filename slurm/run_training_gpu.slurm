#!/bin/bash -l

#SBATCH --job-name=gpu_dvae_training       # Job name
#SBATCH --nodes=1                          # Use one node
#SBATCH --ntasks=1                         # Run a single task
#SBATCH --cpus-per-task=10                 # Use all 36 CPU cores per task to maximize CPU usage
#SBATCH --time=7-00:00:00                  # Time limit set to 7 days
#SBATCH --output=./logs/gpu_training_%j.log # Standard output log
#SBATCH --error=./logs/gpu_training_%j.err  # Standard error log
#SBATCH --partition=gpu                    # Specify the GPU partition
#SBATCH --gres=gpu:2                       # Request 4 GPUs to maximize GPU usage within your allocation
#SBATCH --mem=128G                         # Request the full memory available on a node to avoid memory bottlenecks

# Print the time, hostname, and job ID
echo "Time BEGIN: $(date)"
echo "Running on host: $(hostname)"
echo "Under SLURM JobID: $SLURM_JOBID"

export CUDA_LAUNCH_BLOCKING=1

# Load CUDA module
module load cuda/11.3

# Ensure the Conda initialization script is sourced. This might need adjustment based on your shell and Conda setup.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate research-DVAE

cd ~/workspace/research-DVAE

# Run the command with the appropriate environment
~/miniconda3/envs/research-DVAE/bin/python train_model.py --cfg ./config/sinusoid/cfg_mt_rnn.ini

echo "Time END: $(date)"
