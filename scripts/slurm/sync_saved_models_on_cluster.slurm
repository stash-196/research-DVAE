#!/bin/bash
#SBATCH --job-name=sync_saved_models
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --output=sync_%j.log
#SBATCH --error=sync_%j.err
#SBATCH --partition=compute

# Sync saved models from flash to bucket
FLASH_ROOT="/flash/DoyaU/stash/research-DVAE/saved_model"
BUCKET_ROOT="/bucket/DoyaU/stash/research-DVAE/saved_model"

# List of directories to sync (add more as needed)
DIRS=(
    # 2025-11-12
    # 2025-11-13
    # 2025-11-14
    # 2025-11-15
    # 2026-01-14
    # 2026-01-16
    # 2026-01-18
    # 2026-01-21
    # 2026-01-22
    # 2026-01-24
    # 2026-01-25
    # 2026-01-27
    # 2026-01-28
    # 2026-01-29
    # 2026-01-30
    2026-02-06
    2026-02-12
)

# Sync each directory
for dir in "${DIRS[@]}"; do
    echo "Syncing $dir..."
    rsync -av --progress "$FLASH_ROOT/$dir" "$BUCKET_ROOT/"
    if [ $? -eq 0 ]; then
        echo "Sync of $dir completed successfully."
    else
        echo "Sync of $dir failed!" >&2
    fi
done

echo "All syncs completed."