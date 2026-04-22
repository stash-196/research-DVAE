#!/bin/bash

# Script to generate and submit SLURM jobs for aggregating evaluation results using aggregate_evaluation_results.py
# Edit the experiments dict below to add/remove target directories

# Define experiment runs as a standard array (list).
# Format: "experiment_directory --parameters param1 param2 --filter key=val"
experiments=(
    #
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2024-11-02/ep20000_8alphas_esp50_nanBers_ptf_MT-RNN_SampRatios --parameters sampling_ratio mask_label"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2024-11-01/ep20000_8alphas_esp50_nanBers_ptf_MT-RNN_SampRatios --parameters sampling_ratio mask_label"

        "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch --parameters observation_process sampling_ratio"
        "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch --parameters observation_process sampling_ratio"
    # # 2026-01-25/
    #     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-25/deigo_cluster/20260125_32mem_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100 --parameters observation_process sampling_ratio"
    #     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-25/deigo_cluster/20260125_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100 --parameters observation_process sampling_ratio"

    # # 2026-01-27/
    #     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-27/deigo_cluster/20260126_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_h100 --parameters observation_process sampling_ratio"
    #     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-27/deigo_cluster/20260127_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_hdims --parameters observation_process sampling_ratio"

    # 2026-01-28/
        # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-28/deigo_cluster/20260128_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch3-4_hdims_ptientHigh --parameters dim_rnn sampling_ratio"

    # # 2026-01-29/
    #     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-29/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh --parameters observation_process sampling_ratio"

    # # 2026-01-30/
    #     "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-30/deigo_cluster/20260129_XHRO_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_Subj70_ch1-2_hdi20s_ptientHigh --parameters observation_process sampling_ratio"

    # 2026-01-14/
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch --parameters sampling_ratio --filter loss_mask_mode=none dataset_label=XHRO_02_XH070"
    # used
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch --parameters tag sampling_ratio --filter dataset_label=XHRO_02_XH070"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-01-14/deigo_cluster/20260114_XHRO_ssHIGH-AllLoss_v-MT-RNN_ss_3Subjs_h256_1Dch --parameters observation_process sampling_ratio  --filter dataset_label=XHRO_02_XH070 tag=MT_RNN"
    # 2026-02-12/
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_LSTM_hdi20_ptientHigh --parameters sampling_ratio mask_label"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_epoch10000_len1000_ptfAll_MissAll_clip1_LossNone_MTRNN_hdi20_ptientHigh --parameters sampling_ratio mask_label --filter tag=MT-RNN"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_LSTM_hdi20s_ptientHigh --parameters sampling_ratio mask_label"
    # "/flash/DoyaU/stash/research-DVAE/saved_model/2026-02-12/deigo_cluster/20260212_Lorenz_len1000_drop0_ptf0.6-7-_clip1_AllLoss_MTRNN_hdi20-40_ptientHigh --parameters sampling_ratio mask_label"
)

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# Define paths
CONTAINER_PATH=/bucket/DoyaU/stash/containers/generic_ml_container.sif
PROJECT_PATH=~/workspace/research-DVAE
VENV_PATH=~/containers/venvs/research-DVAE/
DATA_HOST_PATH=/bucket/DoyaU/stash/research-DVAE/data
SAVED_HOST_PATH=/flash/DoyaU/stash/research-DVAE/saved_model

# Loop over each experiment entry in the array
job_index=0
for entry in "${experiments[@]}"; do
    job_index=$((job_index + 1))

    # Extract directory (first word) and arguments (everything else)
    EXPERIMENT_DIR="${entry%% *}"
    FULL_ARGS="${entry#* }"

    if [ -z "$FULL_ARGS" ]; then
        echo "[bash] Skipping entry with empty arguments for directory: $EXPERIMENT_DIR"
        continue
    fi

    if [ ! -d "$EXPERIMENT_DIR" ]; then
        echo "[bash] Skipping missing experiment directory: $EXPERIMENT_DIR"
        continue
    fi

    # Parse parameters and filters from FULL_ARGS
    PARAMETERS=""
    FILTERS=""
    parsing_parameters=false
    parsing_filters=false
    for arg in $FULL_ARGS; do
        if [ "$arg" = "--parameters" ]; then
            parsing_parameters=true
            parsing_filters=false
        elif [ "$arg" = "--filter" ]; then
            parsing_filters=true
            parsing_parameters=false
            FILTERS="$FILTERS --filter"
        elif [ "$parsing_parameters" = true ]; then
            PARAMETERS="$PARAMETERS $arg"
        elif [ "$parsing_filters" = true ]; then
            FILTERS="$FILTERS $arg"
        fi
    done

    # Trim leading spaces
    PARAMETERS=$(echo "$PARAMETERS" | sed 's/^ *//')
    FILTERS=$(echo "$FILTERS" | sed 's/^ *//')

    # Log directory
    LOG_DIR="$EXPERIMENT_DIR/aggregate_logs"
    echo "[bash] Processing experiment: $EXPERIMENT_DIR"
    echo "[bash] Parameters: $PARAMETERS"
    echo "[bash] Filters: $FILTERS"
    echo "[bash] LOG_DIR: $LOG_DIR"
    mkdir -p "$LOG_DIR"

    # Extract base name for the job
    EXPERIMENT_BASENAME=$(basename "$EXPERIMENT_DIR")

    # Compute container paths
    EXPERIMENT_CONTAINER_PATH=${EXPERIMENT_DIR/#$SAVED_HOST_PATH/\/saved_model}

    # Create output directory for plots inside the experiment dir
    OUTPUT_DIR_HOST="${EXPERIMENT_DIR}/aggregated_plots"
    OUTPUT_DIR_CONTAINER="${EXPERIMENT_CONTAINER_PATH}/aggregated_plots"
    mkdir -p "$OUTPUT_DIR_HOST"

    # Temporary SLURM script
    SLURM_SCRIPT="scripts/slurm/temp/run_agg_${EXPERIMENT_BASENAME}_${job_index}.slurm"

    cat > "$SLURM_SCRIPT" <<EOL
#!/bin/bash
#SBATCH --job-name=${EXPERIMENT_BASENAME}_agg_${job_index}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=${LOG_DIR}/%j_agg_${EXPERIMENT_BASENAME}_${job_index}.log
#SBATCH --error=${LOG_DIR}/%j_agg_${EXPERIMENT_BASENAME}_${job_index}.err
#SBATCH --partition=compute

# Define variables
CONTAINER_PATH=$CONTAINER_PATH
PROJECT_PATH=$PROJECT_PATH
VENV_PATH=$VENV_PATH
DATA_HOST_PATH=$DATA_HOST_PATH
SAVED_HOST_PATH=$SAVED_HOST_PATH
EXPERIMENT_DIR=$EXPERIMENT_DIR
LOG_DIR=$LOG_DIR

echo "[slurm] Time BEGIN: \$(date)"
echo "[slurm] Running on host: \$(hostname)"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"
echo "[slurm] Log file: \${LOG_DIR}/%j_agg_${EXPERIMENT_BASENAME}_${job_index}.log"
echo "[slurm] EXPERIMENT_CONTAINER_PATH: $EXPERIMENT_CONTAINER_PATH"
echo "[slurm] Parameters: $PARAMETERS"
echo "[slurm] Filters: $FILTERS"

ml singularity

# Run the Apptainer container
singularity exec \\
  --bind \$PROJECT_PATH:/workspace/project \\
  --bind \$VENV_PATH:/workspace/venv \\
  --bind \$DATA_HOST_PATH:/data \\
  --bind \$SAVED_HOST_PATH:/saved_model \\
  \$CONTAINER_PATH \\
    bash -c "source /workspace/venv/bin/activate && python3 src/dvae/eval/aggregate_evaluation_results.py $EXPERIMENT_CONTAINER_PATH --parameters $PARAMETERS $FILTERS --output_dir $OUTPUT_DIR_CONTAINER"

# Check exit code
EXIT_CODE=\$?
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Error: Job failed with exit code \$EXIT_CODE"
    exit \$EXIT_CODE
fi

echo "[slurm] Time END: \$(date)"
EOL

    # Submit the temporary SLURM script to the queue
    echo "[bash] Submitting aggregation for $EXPERIMENT_BASENAME"
    sbatch "$SLURM_SCRIPT"

done
