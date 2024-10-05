#!/bin/bash

cd ~/workspace/research-DVAE

# Base directory where the configurations are stored
BASE_DIR=~/workspace/research-DVAE/config/sinusoid/generated

# Define a list of experiment names, each corresponding to a subdirectory under BASE_DIR
declare -a experiments=("h64_ep20000_esp30_nanBer_LASTalphas_SampMeth_SampRatio_NoV_0")
# 1" "compare_alphas_&_sampling_methods_2" "compare_alphas_&_sampling_methods_3" "compare_alphas_&_sampling_methods_4" "compare_alphas_&_sampling_methods_5") # Modify this line to include your actual experiment names

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

# Loop over each experiment name to process its configuration files
for experiment in "${experiments[@]}"; do
    CONFIG_DIR="$BASE_DIR/$experiment"



    # Use the find command to locate .ini files and iterate over them
    find "$CONFIG_DIR" -name "*.ini" | while read CONFIG_FILE; do
        # Extract the base name of the configuration file to be used in the job name
        CONFIG_BASENAME=$(basename "$CONFIG_FILE" .ini)

        # Create a unique identifier for the temporary files
        UNIQUE_ID=$(uuidgen)

        # Create a temporary SLURM script for this configuration
        cat > "slurm/temp/run_training_$CONFIG_BASENAME.slurm" <<EOL
#!/bin/bash

#SBATCH --job-name=${CONFIG_BASENAME}_training  # Job name based on config file
#SBATCH --nodes=1                               # Use one node
#SBATCH --ntasks=1                              # Run a single task
#SBATCH --cpus-per-task=8                       # Number of CPU cores per task
#SBATCH --mem=16G                               # Total memory limit
#SBATCH --time=4-00:00:00                       # Time limit hrs:min:sec
#SBATCH --output=${LOG_DIR}/%j_training_${CONFIG_BASENAME}.log  # Standard output log
#SBATCH --error=${LOG_DIR}/%j_training_${CONFIG_BASENAME}.err   # Standard error log
#SBATCH --partition=compute                     # Specify the GPU partition

# Print the time, hostname, and job ID
echo "[slurm] Time BEGIN: \`date\`"
echo "[slurm] Running on host: \`hostname\`"
echo "[slurm] Under SLURM JobID: \$SLURM_JOBID"

# Singularity sif file location
SIF_FILE=~/workspace/research-DVAE/my_container.sif
ml singularity


# Create directories under saved_root and copy params_being_compared.txt
OUTPUT_EXPERIMENT_DIR="${OUTPUT_TODAY_DIR}/${experiment}"
echo "[slurm] OUTPUT_EXPERIMENT_DIR: \$OUTPUT_EXPERIMENT_DIR"
mkdir -p "\$OUTPUT_EXPERIMENT_DIR"
cp "${CONFIG_DIR}/params_being_compared.json" "\$OUTPUT_EXPERIMENT_DIR/"


cd ~/workspace/research-DVAE

echo "[slurm] Running under temp/run_training_$CONFIG_BASENAME.slurm"

# Source Activate  environment, if there is one (e.g., conda or virtualenv)
source pyenv-research-DVAE/bin/activate
# check which python is being used
# echo "[slurm] Using python: `which python`"

echo "[slurm] Running the training script with config file: $CONFIG_FILE, job_id: \$SLURM_JOBID"
# Run the command
singularity exec \$SIF_FILE python3 train_model.py --cfg $CONFIG_FILE --job_id \$SLURM_JOBID

# Print the time again
echo "[slurm] Time END: `date`"


EOL

        # Submit the temporary SLURM script to the queue
        echo "[bash] Submitting $experiment / run_training_$CONFIG_BASENAME.slurm"
        sbatch "slurm/temp/run_training_$CONFIG_BASENAME.slurm"

        # Optionally, remove the temporary SLURM script after submission
        # rm "temp/run_training_$CONFIG_BASENAME.slurm"

    done
done
