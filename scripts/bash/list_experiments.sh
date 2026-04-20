#!/bin/bash

# Script to list experiment directories under saved_model, grouped by date
# Outputs in a format suitable for copying into the experiments array
# Also saves the output to bin/experiment_list.txt

SAVED_MODEL_DIR="/flash/DoyaU/stash/research-DVAE/saved_model"

# Find all date directories (assuming YYYY-MM-DD format)
for date_dir in "$SAVED_MODEL_DIR"/*/; do
    if [ -d "$date_dir" ] && [[ $(basename "$date_dir") =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
        date_name=$(basename "$date_dir")
        echo "$date_name/"

        # Find experiment directories under deigo_cluster
        for exp_dir in "$date_dir"deigo_cluster/*/; do
            if [ -d "$exp_dir" ]; then
                # Remove trailing slash for the path
                exp_path="${exp_dir%/}"
                echo "    \"$exp_path\""
            fi
        done
        echo ""  # Blank line after each date
    fi
done | tee scripts/bash/experiment_list.txt