#!/bin/bash
# Submit one CPU job per GRU-D ini.
#
# Configs come from bin/generate_gru_d_configs.py, not from
# bin/generate_config_file_for_multiple.py. This does not call
# bin/train_model.py, so it does not enter the OTF training loop.
#
#   bash scripts/slurm/train/run_gru_d_multiple.sh gru_d_lorenz_baselines

if [ $# -eq 0 ]; then
    echo "Usage: $0 EXPERIMENT_NAME [EXPERIMENT_NAME ...]" >&2
    echo "Names are directories under config/general_signal/generated/" >&2
    exit 1
fi

BASE_DIR=~/workspace/research-DVAE/config/general_signal/generated
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
    if ! find "$CONFIG_DIR" -name "cfg_gru_d_*.ini" | grep -q .; then
        echo "Error: no cfg_gru_d_*.ini files in $CONFIG_DIR" >&2
        exit 1
    fi
    LOG_DIR="$OUTPUT_TODAY_DIR/deigo_cluster/$experiment/logs"
    mkdir -p "$LOG_DIR"

    find "$CONFIG_DIR" -name "cfg_gru_d_*.ini" | while read CONFIG_FILE; do
        CONFIG_BASENAME=$(basename "$CONFIG_FILE" .ini)
        CONFIG_CONTAINER_PATH=${CONFIG_FILE/#$PROJECT_PATH/\/workspace\/project}

        cat > "$TEMP_DIR/run_gru_d_$CONFIG_BASENAME.slurm" <<EOL
#!/bin/bash
#SBATCH --job-name=grud_${CONFIG_BASENAME}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=${LOG_DIR}/%j_training_${CONFIG_BASENAME}.log
#SBATCH --error=${LOG_DIR}/%j_training_${CONFIG_BASENAME}.err
#SBATCH --partition=compute

CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
CONFIG_FILE=$CONFIG_FILE
LOG_DIR=$LOG_DIR
CONFIG_BASENAME=$CONFIG_BASENAME
CONFIG_CONTAINER_PATH=$CONFIG_CONTAINER_PATH

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
export NUMBA_CACHE_DIR=/tmp/numba_cache_\$SLURM_JOBID
mkdir -p "\$NUMBA_CACHE_DIR"
export MPLBACKEND=Agg

for PATH_VAR in "\$CONTAINER_PATH" "\$PROJECT_PATH" "\$VENV_PATH" "\$DATA_HOST_PATH" "\$SAVED_HOST_PATH"; do
    if [ ! -d "\$PATH_VAR" ] && [ ! -f "\$PATH_VAR" ]; then
        echo "Error: \$PATH_VAR does not exist"
        exit 1
    fi
done

if ! type ml >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile 2>/dev/null || true
fi
if [ -f /etc/profile.d/zz_deigo_base.sh ]; then
  source /etc/profile.d/zz_deigo_base.sh
fi
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
ml singularity

singularity run \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
  bin/train_gru_d.py --job_id \$SLURM_JOBID --cfg \$CONFIG_CONTAINER_PATH

EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code \$EXIT_CODE"
    exit \$EXIT_CODE
fi
echo "[slurm] Time END: \$(date)"
EOL

        echo "[bash] Submitting GRU-D $experiment / $CONFIG_BASENAME"
        sbatch "$TEMP_DIR/run_gru_d_$CONFIG_BASENAME.slurm"
    done
done
