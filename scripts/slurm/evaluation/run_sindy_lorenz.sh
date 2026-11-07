#!/bin/bash
# Fit and score the SINDy Lorenz63 baseline (CPU, one job).
#
# Fully observed xyz, only_x, and an optional masked xyz run. MASK_LABEL
# must match a mask pickle (for example Bernoulli_0.5). Leave it unset to
# skip the masked condition.
#
#   bash scripts/slurm/evaluation/run_sindy_lorenz.sh
#   MASK_LABEL=Bernoulli_0.5 bash scripts/slurm/evaluation/run_sindy_lorenz.sh
#
# This does not call eval_signal or the OTF eval slurm scripts.

set -euo pipefail

CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=/bucket/DoyaU/stash/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model
today=$(date +%Y-%m-%d)
EXPERIMENT=${EXPERIMENT_NAME:-sindy_lorenz_baselines}
LOG_DIR="$SAVED_HOST_PATH/$today/deigo_cluster/$EXPERIMENT/logs"
mkdir -p "$LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/../temp"
mkdir -p "$TEMP_DIR"
SLURM_FILE="$TEMP_DIR/run_sindy_lorenz.slurm"
MASK_LABEL=${MASK_LABEL:-}

cat > "$SLURM_FILE" <<EOL
#!/bin/bash
#SBATCH --job-name=sindy_lorenz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=${LOG_DIR}/%j_sindy_lorenz.log
#SBATCH --error=${LOG_DIR}/%j_sindy_lorenz.err
#SBATCH --partition=compute

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] host=\$(hostname) job=\$SLURM_JOBID"
export NUMBA_CACHE_DIR=/tmp/numba_cache_\$SLURM_JOBID
mkdir -p "\$NUMBA_CACHE_DIR"
export MPLBACKEND=Agg

if ! type ml >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile 2>/dev/null || true
fi
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
ml singularity

run_one() {
  local name="\$1"
  shift
  singularity run \\
    --bind ${PROJECT_PATH}:/workspace/project \\
    --bind ${VENV_PATH}:/workspace/venv \\
    --bind ${DATA_HOST_PATH}:/data \\
    --bind ${SAVED_HOST_PATH}:/saved_model \\
    ${CONTAINER_PATH} \\
    bin/eval_sindy_lorenz.py --output-dir /saved_model/${today}/deigo_cluster/${EXPERIMENT}/\${name} "\$@"
}

run_one all_xyz --observation-process all_xyz --missing-strategy observed_segments
run_one only_x --observation-process only_x --missing-strategy observed_segments
EOL

if [ -n "$MASK_LABEL" ]; then
    cat >> "$SLURM_FILE" <<EOL
run_one all_xyz_mask_${MASK_LABEL} --observation-process all_xyz --mask-label ${MASK_LABEL} --missing-strategy observed_segments
run_one all_xyz_interp_${MASK_LABEL} --observation-process all_xyz --mask-label ${MASK_LABEL} --missing-strategy interpolate
EOL
fi

cat >> "$SLURM_FILE" <<'EOL'
echo "[slurm] Time END: $(date)"
EOL

echo "[bash] Submitting SINDy Lorenz baseline"
sbatch "$SLURM_FILE"
