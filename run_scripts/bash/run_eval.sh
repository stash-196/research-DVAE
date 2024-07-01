#!/bin/bash

# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"

# source activate environment_name
# SIF_FILE = ~/workspace/research-DVAE/my_container.sif

cd ~/workspace/research-DVAE

# Directory to start searching from
search_dir="/Users/stashtomonaga/workspace/research-DVAE/saved_model/2024-06-27"

# Find all files matching *final*.pt recursively starting from search_dir
find "$search_dir" -type f -name "*final*.pt" | while read final_model; do
    # Check if the file exists (redundant check because find only returns existing files)
    if [[ -f "$final_model" ]]; then
        echo "[bash] Running evaluation for model: $final_model"
        # Run the command
        # Running Locally                
        # ~/workspace/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model"
        # Running Locally with nohup
        nohup /Users/stashtomonaga/workspace/research-DVAE/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model" > run_scripts/temp/eval_$
        
        # Running on HPC
        # apptainer exec --nv $SIF_FILE python3 eval_sinusoid.py --saved_dict "$final_model"
    fi
done

# Print the time again
echo "Time END: `date`"
