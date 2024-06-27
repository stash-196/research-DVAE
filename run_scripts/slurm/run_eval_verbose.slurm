#!/bin/bash

#SBATCH --job-name=dvae_eval     # Job name
#SBATCH --nodes=1                    # Use one node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=8            # Number of CPU cores per task
#SBATCH --mem=16G                    # Total memory limit
#SBATCH --time=10:00:00              # Time limit hrs:min:sec
#SBATCH --output=./logs/eval_%j.log  # Standard output and error log
#SBATCH --error=./logs/eval_%j.err   # Standard output and error log
#SBATCH --partition=compute          # Specify the GPU partition

# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"
echo "Under SLURM JobID: $SLURM_JOBID"

# Activate environment, if there is one (e.g., conda or virtualenv)
echo "Activating Conda environment..."
source ~/miniconda3/bin/activate
conda activate research-DVAE
echo "Environment activated."

cd ~/workspace/research-DVAE
echo "Changed directory to: $(pwd)"

# Iterate over all directories in saved_model
saved_model_dir="/flash/DoyaU/stash/research-DVAE/saved_model"
for date_dir in "$saved_model_dir"/*; do
    echo "Checking date directory: $date_dir"
    if [[ -d "$date_dir" ]]; then
        # Iterate over subdirectories in each date directory
        for model_dir in "$date_dir"/*; do
            echo "Checking model directory: $model_dir"
            if [[ -d "$model_dir" ]]; then
                # Check for files containing "final" in their names
                for final_model in "$model_dir"/*final*.pt; do
                    # If such a file exists
                    if [[ -f "$final_model" ]]; then
                        echo "Running evaluation for model: $final_model"
                        # Run the command
                        ~/miniconda3/envs/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model"
                    fi
                done
            fi
        done
    fi
done

echo "All models processed."

# Print the time again
echo "Time END: `date`"
