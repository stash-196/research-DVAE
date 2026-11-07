#!/bin/bash
# Submit one CPU job per filtering-baseline JSON config.
# Usage:
#   bash scripts/slurm/baselines/run_filtering_baselines.sh EXPERIMENT_NAME
# EXPERIMENT_NAME is a directory under config/filtering_baselines/generated/.
# This does not submit OTF training and does not read those generated .ini files.

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 EXPERIMENT_NAME" >&2
    exit 1
fi

EXPERIMENT="$1"
PROJECT_PATH="${PROJECT_PATH:-$HOME/workspace/research-DVAE}"
BASE_DIR="$PROJECT_PATH/config/filtering_baselines/generated/$EXPERIMENT"
CONTAINER_PATH="${CONTAINER_PATH:-/bucket/DoyaU/stash/containers/generic_ml_container.sif}"
VENV_PATH="${VENV_PATH:-/bucket/DoyaU/stash/containers/venvs/research-DVAE/}"
DATA_HOST_PATH="${DATA_HOST_PATH:-/bucket/DoyaU/stash/research-DVAE/data}"
SAVED_HOST_PATH="${SAVED_HOST_PATH:-/flash/DoyaU/stash/research-DVAE/saved_model}"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: no config dir $BASE_DIR" >&2
    exit 1
fi

today=$(date +%Y-%m-%d)
LOG_DIR="$SAVED_HOST_PATH/$today/deigo_cluster/filtering_${EXPERIMENT}/logs"
OUT_HOST="$SAVED_HOST_PATH/$today/deigo_cluster/filtering_${EXPERIMENT}"
mkdir -p "$LOG_DIR" "$OUT_HOST"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/../temp"
mkdir -p "$TEMP_DIR"

find "$BASE_DIR" -name "*.json" | while read -r CONFIG_FILE; do
    CONFIG_BASENAME=$(basename "$CONFIG_FILE" .json)
    CONFIG_CONTAINER_PATH=${CONFIG_FILE/#$PROJECT_PATH/\/workspace\/project}
    OUT_CONTAINER="/saved_model/$today/deigo_cluster/filtering_${EXPERIMENT}/${CONFIG_BASENAME}"
    cat > "$TEMP_DIR/run_filtering_${CONFIG_BASENAME}.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=filt_${CONFIG_BASENAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=${LOG_DIR}/%j_filtering_${CONFIG_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_filtering_${CONFIG_BASENAME}.err
#SBATCH --partition=compute

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] JobID: \$SLURM_JOBID"

if ! type ml >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
fi
ml singularity

singularity exec \\
  --bind ${PROJECT_PATH}:/workspace/project \\
  --bind ${VENV_PATH}:/workspace/venv \\
  --bind ${DATA_HOST_PATH}:/data \\
  --bind ${SAVED_HOST_PATH}:/saved_model \\
  ${CONTAINER_PATH} \\
  bash -lc "source /workspace/venv/bin/activate && cd /workspace/project && python bin/fit_eval_filtering_baselines.py --config ${CONFIG_CONTAINER_PATH} --out-dir ${OUT_CONTAINER}"

echo "[slurm] Time END: \$(date)"
EOL
    echo "[bash] Submitting $CONFIG_BASENAME"
    sbatch "$TEMP_DIR/run_filtering_${CONFIG_BASENAME}.slurm"
done
