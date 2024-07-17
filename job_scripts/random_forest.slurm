#!/bin/bash
#SBATCH --job-name="random_forest"
#SBATCH --output="../outputs/random_forest_%j_%N.out"
#SBATCH --mem=32g
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuA40x4        # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --account=bbug-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=02:00:00             # hh:mm:ss for the job
#SBATCH

export epochs=1
export time_step=36
export data_path=''
export output_path=''
export mode='random_forest'

now="$(date +"%T")"
echo "job is starting on $HOSTNAME ($now)"

# Set up env
source activate CropTS
    
python  \
    --data_path $data_path \
    --output_path $output_path \
    --mode $mode \
    --seq_len $time_step \
    --test_name random_forest_${time_step}steps \
    --agg concat
    
now="$(date +"%T")"
echo "job complete on $HOSTNAME ($now)"
exit