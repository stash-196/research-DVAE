#!/bin/bash

# Sync saved models from flash to bucket (simple bash script)
FLASH_ROOT="/flash/DoyaU/stash/research-DVAE/saved_model"
BUCKET_ROOT="/bucket/DoyaU/stash/research-DVAE/saved_model"

# List of directories to sync (add more as needed)
DIRS=(
    25-05-21
    2025-05-22
    2025-05-23
    2025-06-15
    2025-06-20
    2025-06-21
    2025-06-28
    2025-07-09
    2025-09-02
    2025-09-03
    2025-10-15
    2025-10-16
    2025-10-18
    2025-10-19
    2025-10-20
    2025-10-22
    2025-10-24
    2025-10-25
    2025-11-12
    2025-11-13
    2025-11-14
    2025-11-15
    2026-01-14
    2026-01-16
    2026-01-18
    2026-01-21
    # 2026-01-22
    # 2026-01-24
    # 2026-01-25
    # 2026-01-27
    # 2026-01-28
    # 2026-01-29
    # 2026-01-30
    # 2026-02-06
    # 2026-02-12
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