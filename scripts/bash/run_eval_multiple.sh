#!/bin/bash

# Print the time, hostname, and job ID
echo "Time BEGIN: `date`"
echo "Running on host: `hostname`"

# source activate environment_name
SIF_FILE=~/containers/dvae_cuda11.8_py3.10.sif
echo "[bash] Using Singularity Image: $SIF_FILE"

cd ~/workspace/research-DVAE

# Directory to start searching from
# search_dir="/Users/stashtomonaga/workspace/research-DVAE/saved_model/2024-06-27"
search_dir="/home/stash/storage/research-DVAE/saved_model"
echo "[bash] Searching for models in directory: $search_dir"

# Find all files matching *final*.pt recursively starting from search_dir
find "$search_dir" -type f -name "*final*.pt" | while read final_model; do
    # Check if the file exists (redundant check because find only returns existing files)
    if [[ -f "$final_model" ]]; then
        echo "[bash] Running evaluation for model: $final_model"
        # Name of directory containing the $final_model, without the whole path
        model_dirname=$(basename $(dirname "$final_model"))
        log_file=run_scripts/temp/eval_$model_dirname.log
        echo "[bash] Log file: $log_file"
        # Run the command
        # Running Locally                
        # ~/workspace/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model"
        # Running Locally with nohup
        # nohup /Users/stashtomonaga/workspace/research-DVAE/research-DVAE/bin/python eval_sinusoid.py --saved_dict "$final_model" > run_scripts/temp/eval_$
        
        # Running on HPC
        # apptainer exec --nv $SIF_FILE python3 eval_sinusoid.py --saved_dict "$final_model"
        # Running on HPC with nohup
        nohup apptainer exec --nv $SIF_FILE python3 eval_sinusoid.py --saved_dict "$final_model" > $log_file &
        

    fi
done

# Print the time again
echo "Time END: `date`"
