#!/bin/bash

echo "[bash]Time BEGIN: `date`"
echo "[bash]Running on host: `hostname`"
echo "[bash]Under SLURM JobID: $SLURM_JOBID"


cd ~/workspace/research-DVAE
echo "[bash]Changed directory to: $(pwd)"

echo "[bash]Activating Conda environment..."
source research-DVAE/bin/activate
echo "[bash]Environment activated."


saved_model_dir="saved_model/2024-05-15"

# Read all date directories into an array

# Get the length of the array
num_dirs=${#date_dirs[@]}

# Iterate over date directories in reverse order



for experiment_dir in "$saved_model_dir"/*; do
    echo "[bash]Checking experiment directory: $experiment_dir"
    for model_dir in "$experiment_dir"/*; do
        echo "[bash]Checking model directory: $model_dir"
        if [[ -d "$model_dir" ]]; then
            # Check for files containing "final" in their names
            final_models=("$model_dir"/*final*.pt)
            for final_model in "${final_models[@]}"; do
                # If such a file exists
                if [[ -f "$final_model" ]]; then
                    echo "[bash]Running evaluation for model: $final_model"
                    # Run the command
                    research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model"
                fi
            done
        fi
    done
done


echo "[bash]All models processed."
echo "[bash]Time END: `date`"
