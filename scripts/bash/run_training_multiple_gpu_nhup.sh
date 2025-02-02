#!/bin/bash

# Base directory where the configurations are stored
BASE_DIR=~/workspace/research-DVAE/config/sinusoid/generated

# Define a list of experiment names, each corresponding to a subdirectory under BASE_DIR
declare -a experiments=("h90_ep1000_esp10_SampMeths_RNNs_0")

# Get the current date in YYYY-MM-DD format
today=$(date +%Y-%m-%d)

# get the current time in HH:MM:SS:MS format
now=$(date +%H:%M:%S) 

# Parse cfg_device.ini to get the saved_root path
CFG_DEVICE_FILE=~/workspace/research-DVAE/config/cfg_device.ini
SAVE_ROOT=$(sed -n 's/^saved_root\s*=\s*//p' $CFG_DEVICE_FILE)
echo "[bash] SAVE_ROOT: $SAVE_ROOT"

OUTPUT_TODAY_DIR="$SAVE_ROOT/$today"
echo "[bash] OUTPUT_TODAY_DIR: $OUTPUT_TODAY_DIR"

# Directory for storing logs
LOG_DIR=$OUTPUT_TODAY_DIR/$experiments/logs
echo "[bash] LOG_DIR: $LOG_DIR"
mkdir -p "$LOG_DIR"
ls $SAVE_ROOT

# Loop over each experiment name to process its configuration files
for experiment in "${experiments[@]}"; do
    CONFIG_DIR="$BASE_DIR/$experiment"

    # Use the find command to locate .ini files and iterate over them
    find "$CONFIG_DIR" -name "*.ini" | while read CONFIG_FILE; do
        # Extract the base name of the configuration file to be used in the job name
        CONFIG_BASENAME=$(basename "$CONFIG_FILE" .ini)

        # Create directories under saved_root and copy params_being_compared.txt
        OUTPUT_EXPERIMENT_DIR="${OUTPUT_TODAY_DIR}/${experiment}"
        echo "[bash] OUTPUT_EXPERIMENT_DIR: $OUTPUT_EXPERIMENT_DIR"
        mkdir -p "$OUTPUT_EXPERIMENT_DIR"
        cp "${CONFIG_DIR}/params_being_compared.json" "$OUTPUT_EXPERIMENT_DIR/"

        # Log files
        LOG_FILE=${LOG_DIR}/${CONFIG_BASENAME}_training.log
        ERR_FILE=${LOG_DIR}/${CONFIG_BASENAME}_training.err

        # Singularity sif file location
        SIF_FILE=~/containers/dvae_cuda11.8_py3.10.sif

        # Run the command using nohup and &
        echo "[bash] Running the training script with config file: $CONFIG_FILE"
        nohup apptainer exec --nv $SIF_FILE python3 train_model.py --cfg $CONFIG_FILE > $LOG_FILE 2> $ERR_FILE &

        # Optionally, you can store the PID of the process if you need to manage it later
        echo $! >> $OUTPUT_TODAY_DIR/running_jobs.pids

    done
done
