#!/bin/bash

# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"

# source activate environment_name
$SIF_

cd ~/workspace/research-DVAE

# Iterate over all directories in saved_model
for model_dir in /Users/stashtomonaga/workspace/research-DVAE/saved_model/results/
    # If it is a directory
    if [[ -d "$model_dir" ]]; then
        # if [[ "${model_dir##*/}" == *Sinusoid* ]]; then
            # Check for files containing "final" in their names
            for final_model in "$model_dir"/*final*.pt; do
                # If such a file exists
                if [[ -f "$final_model" ]]; then
                    # Run the command
                    ~/miniconda3/envs/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model"
                fi
            done
        # fi
    fi
done

# Print the time again
echo "Time END: `date`"
