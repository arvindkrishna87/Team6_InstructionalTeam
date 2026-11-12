#!/bin/bash
# Test B: UNI2 features from each patch at its original size. Each array task encodes every 4th patch.
# Rerunning skips patches already saved.
#SBATCH --account=e32998
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-3
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --job-name=SAME_INPUT_B_embed
#SBATCH --output=/home/ele0436/STAT390_coder/same_input_comparison/logs/%x_%A_%a.out
#SBATCH --error=/home/ele0436/STAT390_coder/same_input_comparison/logs/%x_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ele0436@u.northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env
export HF_HUB_OFFLINE=1

X=/home/ele0436/STAT390_coder/same_input_comparison
cd $X/code
python3 precompute_pooled_features.py \
    --encoder uni2 --uni2_native_size \
    --patches_dir /projects/e32998/patches_benign_split \
    --embeddings_dir $X/embeddings_uni2_original_size \
    --batch_size 64 --num_workers 6 \
    --num_shards 4 --shard_index "$SLURM_ARRAY_TASK_ID" --log_every 20
