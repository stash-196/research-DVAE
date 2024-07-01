#!/bin/bash

# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"

# source activate environment_name
# SIF_FILE = ~/workspace/research-DVAE/my_container.sif

cd ~/workspace/research-DVAE

# Directory to start searching from
saved_model_dir="/Users/stashtomonaga/workspace/research-DVAE/saved_model/"

for date_dir in "$saved_model_dir"/*; do
    echo "[bash] Checking date directory: $date_dir"
    if [[ -d "$date_dir" ]]; then # Check if the date directory exists
        # Iterate over subdirectories in each date directory
        for exp_dir in "$date_dir"/*; do
            echo "[bash] Checking experiment directory: $exp_dir"
            /Users/stashtomonaga/workspace/research-DVAE/research-DVAE/bin/python aggregate_and_visualize_metrics2.py --exp_dir "$exp_dir"        
        done
    fi
done

# Print the time again
echo "Time END: `date`"
