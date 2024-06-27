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

# Source Activate  environment, if there is one (e.g., conda or virtualenv)
source pyenv-research-DVAE/bin/activate


# Run the command
pyenv-research-DVAE/bin/python train_model.py --cfg /home/s/sutashu-tomonaga1/workspace/research-DVAE/config/sinusoid/generated/h180_ep500_SampMeths_αs_2/cfg_h180_ep500_SampMeths_αs_2_VRNN_sampling_method-1.ini --job_id $SLURM_JOBID

# Print the time again
echo "Time END: `date`"

