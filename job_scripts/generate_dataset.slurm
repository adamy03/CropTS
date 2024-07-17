#!/bin/bash
#SBATCH --job-name="s2_generation"
#SBATCH --output="s2_generation_%j_%N.out"
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --account=bbug-delta-cpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --partition=cpu       # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --time=02:00:00             # hh:mm:ss for the job
#SBATCH

echo "job is starting on $HOSTNAME"

# Set up env
source activate CropTS

# Exec script
python generate_fused_dataset.py

echo "job complete on $HOSTNAME"