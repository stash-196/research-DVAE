#!/bin/bash
# Script to submit a SLURM job for collecting .ini run statuses using collect_ini_runs.py

# Define the dates you want to process here.
# You can add or remove dates from this array.
experiment_dates=(
    "2026-01-14"
    "2026-01-16"
    "2026-01-18"
    "2026-01-21"
    "2026-01-22"
    "2026-01-24"
    "2026-01-25"
    "2026-01-27"
    "2026-01-28"
    "2026-01-29"
    "2026-01-30"
    "2026-02-06"
    "2026-02-12"
    "2026-04-21"
    "2026-04-24"
    "2026-04-26"
    "2026-05-20"
    "2026-05-21"
    "2026-05-23"
    "2026-05-25"
    "2026-05-26"
    "2026-05-27"
    "2026-05-28"
)

# Paths based on your existing structure
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
INPUT_MODELS_PATH=/flash/DoyaU/stash/research-DVAE/saved_model
OUTPUT_PATH=/flash/DoyaU/stash/research-DVAE/saved_model/collected_experiment_runs_summary

mkdir -p "$OUTPUT_PATH"
LOG_DIR="${OUTPUT_PATH}/logs"
mkdir -p "$LOG_DIR"

# Convert bash array to a space-separated string for the python script
DATES_STR="${experiment_dates[*]}"

# Temporary SLURM script
SLURM_SCRIPT="scripts/slurm/temp/run_collect_ini.slurm"
mkdir -p scripts/slurm/temp

cat > "$SLURM_SCRIPT" <<INNER_EOL
#!/bin/bash
#SBATCH --job-name=collect_ini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=${LOG_DIR}/%j_collect_ini.log
#SBATCH --error=${LOG_DIR}/%j_collect_ini.err
#SBATCH --partition=compute

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"

ml singularity

# Run the Apptainer container
singularity exec \\
  --bind ${PROJECT_PATH}:/workspace/project \\
  --bind ${VENV_PATH}:/workspace/venv \\
  --bind ${DATA_HOST_PATH}:/data \\
  --bind ${INPUT_MODELS_PATH}:/input_models \\
  --bind ${OUTPUT_PATH}:/output_data \\
  ${CONTAINER_PATH} \\
    bash -c "source /workspace/venv/bin/activate && python3 /workspace/project/scripts/slurm/track_experiments/collect_ini_runs.py --dates ${DATES_STR} --saved-model-root /input_models --output /output_data/experiment_runs_summary.csv"

EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code \$EXIT_CODE"
    exit \$EXIT_CODE
fi

echo "[slurm] Time END: \$(date)"
INNER_EOL

echo "============================================================"
echo "Submitting job to collect .ini configs across dates: ${DATES_STR}"
echo "------------------------------------------------------------"
echo "Reading .ini files from: ${INPUT_MODELS_PATH}"
echo "Writing output CSV to:   ${OUTPUT_PATH}/experiment_runs_summary.csv"
echo "Writing SLURM logs to:   ${LOG_DIR}"
echo "============================================================"
sbatch "$SLURM_SCRIPT"
