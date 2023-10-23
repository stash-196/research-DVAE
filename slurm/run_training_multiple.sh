#!/bin/bash

# Define the directory containing the configuration files
CONFIG_DIR=~/workspace/research-DVAE/config/sinusoid/generated4

# Use the find command to locate .ini files and iterate over them
find "$CONFIG_DIR" -name "*.ini" | while read CONFIG_FILE; do
  # Extract the base name of the configuration file to be used in the job name
  CONFIG_BASENAME=$(basename "$CONFIG_FILE" .ini)
  
  # Create a temporary SLURM script for this configuration
  cat > "run_training_$CONFIG_BASENAME.slurm" <<EOL
#!/bin/bash

#SBATCH --job-name=${CONFIG_BASENAME}_training  # Job name based on config file
#SBATCH --nodes=1                               # Use one node
#SBATCH --ntasks=1                              # Run a single task
#SBATCH --cpus-per-task=8                       # Number of CPU cores per task
#SBATCH --mem=16G                               # Total memory limit
#SBATCH --time=10:00:00                         # Time limit hrs:min:sec
#SBATCH --output=./logs/training_${CONFIG_BASENAME}_%j.log  # Standard output log
#SBATCH --error=./logs/training_${CONFIG_BASENAME}_%j.err   # Standard error log
#SBATCH --partition=compute                     # Specify the GPU partition

# Print the time, hostname, and job ID
echo "Time BEGIN: \`date\`"
echo "Running on host: \`hostname\`"
echo "Under SLURM JobID: $SLURM_JOBID"

# Activate the environment (assuming conda)
source ~/miniconda3/bin/activate research-DVAE

cd ~/workspace/research-DVAE

# Run the command with the current configuration file
~/miniconda3/envs/research-DVAE/bin/python train_model.py --cfg "$CONFIG_FILE"

# Print the time again
echo "Time END: \`date\`"

EOL

  # Submit the temporary SLURM script to the queue
  sbatch "run_training_$CONFIG_BASENAME.slurm"

  # Optionally, remove the temporary SLURM script after submission
  rm "run_training_$CONFIG_BASENAME.slurm"

done
