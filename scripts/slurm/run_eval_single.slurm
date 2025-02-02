#!/bin/bash

#SBATCH --job-name=eval_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00  # Adjust based on expected run time per model
#SBATCH --output=./logs/eval_%j.log
#SBATCH --error=./logs/eval_%j.err
#SBATCH --partition=compute

echo "Time BEGIN: $(date)"
echo "Running on host: $(hostname)"
echo "Under SLURM JobID: $SLURM_JOBID"

# Activate the environment
echo "Activating Conda environment..."
source ~/miniconda3/bin/activate
conda activate research-DVAE
echo "Environment activated."

# Navigate to the working directory
cd ~/workspace/research-DVAE
echo "Changed directory to: $(pwd)"

# The argument is the path to the model
final_model=$1
echo "Running evaluation for model: $final_model"
~/miniconda3/envs/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model"

echo "Time END: $(date)"
