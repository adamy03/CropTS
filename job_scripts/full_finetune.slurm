#!/bin/bash
#SBATCH --job-name="full_finetuning"
#SBATCH --output="../outputs/full_finetunting_%j_%N.out"
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuA40x4        # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=bbug-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --time=06:00:00             # hh:mm:ss for the job
#SBATCH

export epochs=30
export reduction='concat'
export init_lr=0.0000001
export max_lr=0.00008
export seq_len=36
export mode='full_finetuning'
export batch_size=12
export data_path=''
export output_path=''
export bands=7

now="$(date +"%T")"
echo "job is starting on $HOSTNAME ($now)"

# Set up env
source activate CropTS

accelerate launch --config_file  \
     \
    --data_path $data_path \
    --output_path $output_path \
    --mode $mode \
    --batch_size $batch_size \
    --init_lr $init_lr \
    --max_lr $max_lr \
    --seq_len $seq_len \
    --reduction $reduction \
    --bands $bands \
    --epochs $epochs \
    --test_name benchmarkv2
    # --test_name ${mode}_${epochs}_${seq_len} \
    # --test_name ${mode}_epochs_${epochs}_lr_${init_lr}_seq_len_${seq_len}_${reduction}_bs_${batch_size}_sar_only \

now="$(date +"%T")"
echo "job complete on $HOSTNAME ($now)"
exit