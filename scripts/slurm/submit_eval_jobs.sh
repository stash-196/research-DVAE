#!/bin/bash

# Define the base directory for saved models
saved_model_dir="/flash/DoyaU/stash/research-DVAE/saved_model"

# Create a unique directory for this run's logs based on the current date and time
log_dir="./logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p $log_dir

for date_dir in "$saved_model_dir"/*; do
    echo "Checking date directory: $date_dir"
    if [[ -d "$date_dir" ]]; then
        for model_dir in "$date_dir"/*; do
            echo "Checking model directory: $model_dir"
            if [[ -d "$model_dir" ]]; then
                for final_model in "$model_dir"/*final*.pt; do
                    if [[ -f "$final_model" ]]; then
                        # Submit a job for each final model, specifying the log directory
                        echo "Submitting evaluation for model: $final_model"
                        sbatch --output=$log_dir/eval_%j.log --error=$log_dir/eval_%j.err run_eval_single.slurm "$final_model"
                    fi
                done
            fi
        done
    fi
done
