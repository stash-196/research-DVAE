#!/bin/bash
# Submit one CPU sbatch job per Professor Forcing ini.
#
#   bash scripts/slurm/baselines/run_professor_forcing.sh professor_forcing_lorenz63
#
# Configs are produced by bin/generate_professor_forcing_configs.py and live
# under config/lorenz63/generated_professor_forcing/<experiment>/.

if [ $# -eq 0 ]; then
    echo "Usage: $0 EXPERIMENT_NAME [EXPERIMENT_NAME ...]" >&2
    echo "Names are directories under config/lorenz63/generated_professor_forcing/" >&2
    exit 1
fi

BASE_DIR=~/workspace/research-DVAE/config/lorenz63/generated_professor_forcing
experiments=("$@")
today=$(date +%Y-%m-%d)

CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=/bucket/DoyaU/stash/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

OUTPUT_TODAY_DIR="$SAVED_HOST_PATH/$today"
mkdir -p "$OUTPUT_TODAY_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_DIR="$SCRIPT_DIR/../temp"
mkdir -p "$TEMP_DIR"

for experiment in "${experiments[@]}"; do
    CONFIG_DIR="$BASE_DIR/$experiment"
    if [ ! -d "$CONFIG_DIR" ]; then
        echo "Error: no config dir $CONFIG_DIR" >&2
        exit 1
    fi
    if ! find "$CONFIG_DIR" -name "*.ini" | grep -q .; then
        echo "Error: no .ini files in $CONFIG_DIR" >&2
        exit 1
    fi
    LOG_DIR="$OUTPUT_TODAY_DIR/deigo_cluster/professor_forcing/$experiment/logs"
    mkdir -p "$LOG_DIR"

    find "$CONFIG_DIR" -name "*.ini" | while read -r CONFIG_FILE; do
        CONFIG_BASENAME=$(basename "$CONFIG_FILE" .ini)
        CONFIG_CONTAINER_PATH=${CONFIG_FILE/#$PROJECT_PATH/\/workspace\/project}

        cat > "$TEMP_DIR/run_professor_forcing_$CONFIG_BASENAME.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=pf_${CONFIG_BASENAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=${LOG_DIR}/%j_pf_${CONFIG_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_pf_${CONFIG_BASENAME}.err
#SBATCH --partition=compute

CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
CONFIG_CONTAINER_PATH=$CONFIG_CONTAINER_PATH

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"

if ! type ml >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile 2>/dev/null || true
fi
if [ -f /etc/profile.d/zz_deigo_base.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/zz_deigo_base.sh
fi
if [ -f /etc/profile.d/modules.sh ]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi
ml singularity

singularity run \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
  bin/train_professor_forcing.py --job_id \$SLURM_JOBID --cfg \$CONFIG_CONTAINER_PATH

EXIT_CODE=\$?
echo "[slurm] Time END: \$(date)"
exit \$EXIT_CODE
EOL
        echo "[bash] Submitting $experiment / $CONFIG_BASENAME"
        sbatch "$TEMP_DIR/run_professor_forcing_$CONFIG_BASENAME.slurm"
    done
done
