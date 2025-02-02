#!/bin/bash

#SBATCH --job-name=eval_model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=5:00:00  # Adjust based on expected run time per model
#SBATCH --output=./logs/eval_multiple_%j.log
#SBATCH --error=./logs/eval_multiple_%j.err
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
echo "[slurm] Changed directory to: $(pwd)"

# source activate environment_name
SIF_FILE=./my_container.sif
echo "[slurm] Using Singularity Image: $SIF_FILE"

# Directory to start searching from
# search_dir="/Users/stashtomonaga/workspace/research-DVAE/saved_model/2024-06-27"
search_dir="/flash/DoyaU/stash/research-DVAE/saved_model/2024-07-01"

# Find all files matching *final*.pt recursively starting from search_dir
find "$search_dir" -type f -name "*final*.pt" | while read final_model; do
    # Check if the file exists (redundant check because find only returns existing files)
    if [[ -f "$final_model" ]]; then
        echo "[slurm] Running evaluation for model: $final_model"
        # Name of directory containing the $final_model, without the whole path
        model_dirname=$(basename $(dirname "$final_model"))
        log_file=run_scripts/temp/eval_$model_dirname.log
        echo "[slurm] Log file: $log_file"

        # Running on HPC with nohup
        ml singularity
        nohup singularity exec --nv $SIF_FILE python3 eval_sinusoid.py --saved_dict "$final_model" > $log_file &
        

    fi
done

# Print the time again
echo "Time END: `date`"

