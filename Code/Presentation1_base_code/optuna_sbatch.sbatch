#!/bin/bash
#SBATCH --account=e32998
#SBATCH --partition=long
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem=128G
#SBATCH --job-name=OPTUNA_5FOLD_MIL
#SBATCH --output=/projects/e32998/...../logs/%x_%j.out
#SBATCH --error=/projects/e32998/....../logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your-email@northwestern.edu

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Host: $(hostname)"

python3 -m pip install -r /projects/e32998/......./requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd # ........your director name.......

python3 optuna_tuning.py \
    --study_name mil_5fold_sobol16_tpe64 \
    --output_dir /projects/e32998/MIL_training/optuna_5fold_runs \
    --storage sqlite:////projects/e32998/MIL_training/optuna_5fold_runs/optuna_5fold.db \
    --splits_dir /projects/e32998/STAT390_Krish/Code/Code9_no_leakage/no_train_val_leakage_splits \
    --splits_pattern 'data_splits_new_0*.npz' \
    --n_trials 64 \
    --sobol_startup_trials 16 \
    --epochs 30 \
    --pruning_warmup_epochs 6 \
    --no_prune_first_trials 6 \
    --batch_size 1 \
    --num_workers 2 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --seed 42

echo "End time: $(date)"
