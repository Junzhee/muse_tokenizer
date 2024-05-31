#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --output=logs/out.%j
#SBATCH --error=error_logs/err.%j
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:4
#SBATCH --partition=booster
#SBATCH --cpus-per-task=96

echo $(nproc) 

export CUDA_VISIBLE_DEVICES=0,1,2,3 


# Interval: 1400
# 5.22: 0010000 - 0029600 not complete due to bug
# 5.23 testing: 0029600 - 0030000 - 1h
# Interval: 2000
# 5.23 run: 0030000
# 5.26 submitted: 0010000 - 0040000 - completed: 0040001 - 0050000 (to check)
# Note we put 20-30 30-40k in the same folder, rmb to seperate after running
# Checked: 0010000 - 0030000
 

START_SHARD="0018001"
echo START_SHARD=$START_SHARD

END_SHARD="0020000"
echo END_SHARD=$END_SHARD

NUM_WORKERS=1
echo NUM_WORKERS=$NUM_WORKERS

NUM_GPUS=4
echo NUM_GPUS=$NUM_GPUS

BATCH_SIZE=128
echo BATCH_SIZE=$BATCH_SIZE

OUTPUT_DIR="/p/scratch/ccstdl/xu17/jz/muse_tokenizer/output_dir/10-20k"
echo OUTPUT_DIR=$OUTPUT_DIR

# # Args
# module load Stages/2023 GCC/11.3.0  OpenMPI/4.1.4
# ml git

# source /p/project/ccstdl/xu17/miniconda3/bin/activate
source /p/scratch/ccstdl/xu17/miniconda3/bin/activate muse
# conda activate muse

srun --cpu-bind=v --accel-bind=gn python /p/scratch/ccstdl/xu17/jz/muse_tokenizer/tokenize_images_datacomp.py \
                                        --batch_size $BATCH_SIZE \
                                        --start_shard $START_SHARD \
                                        --end_shard $END_SHARD \
                                        --num_workers $NUM_WORKERS \
                                        --num_gpus $NUM_GPUS \
                                        --save_file $OUTPUT_DIR

# python -u : produce output immediately, no buffer caching
#srun --cpu-bind=v --accel-bind=gn  python -u dummy_script.py
