#!/bin/bash
#SBATCH --account=e32998                  # Use your Quest project account (check with 'groups')
#SBATCH --partition=long                  # 'short', 'normal', or 'long' partition
#SBATCH --time=40:00:00                    # max run time
#SBATCH --nodes=1                         # 1 node
#SBATCH --ntasks=16                        # n nores
#SBATCH --mem=128G                          # n GB RAM
#SBATCH --job-name=MIL_train              # Job name (no spaces)
#SBATCH --output=test_split5_Oct10.out
#SBATCH --error=test_split5_Oct10.err
#SBATCH --mail-type=ALL                   # Send email on begin, end, and fail
#SBATCH --mail-user=krish@northwestern.edu  # Your NU email

module purge
module load python-anaconda3/2019.10
eval "$(conda shell.bash hook)"
conda activate /projects/e32998/env   # you might want to copy the env and put it in your new project directory, if different from e32998

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Run training
python3 -m pip install -r /projects/e32998/STAT390_Krish/Code/Code9_no_leakage/requirements.txt
python3 -m pip install --upgrade certifi
export SSL_CERT_FILE="$(python -m certifi)"

cd /projects/e32998/STAT390_Krish/Code/Code9_no_leakage/

# without loading checkpoints, trying 5 diff train / test splits

python3 main.py \
    --analyze_attention \
    --attention_top_n 8 \
    --per_slice_cap 500 \
    --max_slices_per_stain 5 \
    --load_splits /projects/e32998/STAT390_Krish/Code/Code9_no_leakage/no_train_val_leakage_splits/data_splits_new_05.npz \
