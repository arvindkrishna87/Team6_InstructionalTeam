This is an update on Code1_5_ModelTraining_Feb18_NoVarPatches_Ben
KimiaNet weights are loaded correctly in this code; models.py updated


# Hierarchical Attention MIL Trainer

## Overview

This implementation converts the original Jupyter notebook into a modular Python structure suitable for Quest. The model uses three levels of attention:

1. **Patch-level attention**: Within each stain-slice
2. **Stain-level attention**: Across slices within each stain  
3. **Case-level attention**: Across different stains (H&E, Melan, SOX10)


## Usage

### Training with Attention Analysis
```bash
python main.py --analyze_attention --attention_top_n 5
```

### Evaluation Only
```bash
python main.py --eval_only --resume /path/to/checkpoint.pth --analyze_attention
```

### Using Existing Data Splits
```bash
python main.py --load_splits ./runs/run_20241028_143022/data_splits.npz
```

### Advanced Options
```bash
python main.py \
    --epochs 10 \
    --lr 1e-4 \
    --embed_dim 512 \
    --per_slice_cap 800 \
    --max_slices_per_stain 5 \
    --analyze_attention \
    --attention_top_n 10 \
    --load_splits /path/to/data_splits.npz \
    --eval_only \
    --resume /path/to/checkpoint.pth \
    --batch_size 32 \
    --weight_decay 1e-5 \
    --dropout 0.1 \
    --patience 5 \
    --min_delta 0.001
```

**Key Arguments:**
You can specify any combination of these arguments to override the defaults in `config.py`:
- `--epochs`: Number of training epochs (default: 30)
- `--lr`: Learning rate (default: 3e-4)
- `--embed_dim`: Embedding dimension (default: 512)
- `--per_slice_cap`: Max patches per slice (default: 800)
- `--max_slices_per_stain`: Max slices per stain (default: None)
- `--analyze_attention`: Enable attention analysis and visualization
- `--attention_top_n`: Number of top/bottom patches to visualize (default: 5)
- `--load_splits`: Load existing train/val/test splits from .npz file
- `--eval_only`: Skip training, only evaluate
- `--resume`: Resume from checkpoint
- `--batch_size`: Batch size (default: 1)
- `--weight_decay`: L2 regularization (default: 2e-4)
- `--seed`: Random seed (default: 42)

**Output Directory:**
The output directory is automatically created with a timestamp in the format `.../run_YYYYMMDD_HHMMSS/` under a base directory, which can be changed by modifying `DATA_PATHS['runs_dir']` in `config.py` (currently set to `/projects/e32998/MIL_training/final_runs`).

## SLURM Job Management

### Submitting Jobs

**Before submitting:**
1. Update the email in the sbatch file: Replace `YOUR_NETID@u.northwestern.edu` with your actual NetID
2. Verify the account and partition settings match your Quest allocation
3. Modify logging directory if needed: Logs are currently set to `/projects/e32998/MIL_training/logs/` in the sbatch files

The `sbatch_files/` directory contains pre-configured SLURM scripts for running training with different non-overlapping data splits:

```bash
# Submit a single job for one split
sbatch sbatch_files/run_training_1st_splits.sbatch

# Or submit all 5 splits for cross-validated results
sbatch sbatch_files/run_training_1st_splits.sbatch
sbatch sbatch_files/run_training_2nd_splits.sbatch
sbatch sbatch_files/run_training_3rd_splits.sbatch
sbatch sbatch_files/run_training_4th_splits.sbatch
sbatch sbatch_files/run_training_5th_splits.sbatch
```

**Training Strategy:**
- Use **one split** for initial experimentation or single model training
- Use **all 5 splits** for robust cross-validated results and performance evaluation

When you submit a job, SLURM will return a job ID (e.g., "Submitted batch job 122345"). You can also get job IDs later using `squeue --me`.

### Monitoring Jobs
```bash
# Check job status
squeue --me

# View real-time log output (replace <JOB_ID> with actual job ID from submission)
tail -f /projects/e32998/MIL_training/logs/training_logs_<JOB_ID>.log

# Cancel a job if needed (replace <JOB_ID> with actual job ID)
scancel <JOB_ID>
```

**Note:** Replace `<JOB_ID>` with the actual job ID returned by SLURM when you submit (e.g., if SLURM says "Submitted batch job 122345", use 122345 as the job ID).

### Job Output
- **SLURM logs** are written to `/projects/e32998/MIL_training/logs/training_logs_<JOB_ID>.log` (configured in sbatch files)
- **Training results** are saved in `/projects/e32998/MIL_training/final_runs/run_YYYYMMDD_HHMMSS/` (configured in config.py)

## Output Structure

Each run creates a timestamped directory with all results:

```
./runs/run_YYYYMMDD_HHMMSS/
├── results.txt                    # Summary metrics
├── predictions.csv                # Per-case predictions
├── confusion_matrix.png           # Visual confusion matrix
├── data_splits.npz                # Case IDs for reproducibility
├── checkpoints/                   # Model weights
│   └── *.pth                      # If not --eval_only
└── attention_analysis/            # (if --analyze_attention)
    ├── attention_summary.txt
    ├── stain_attention_distribution.png
    └── patch_attention/
        ├── case_*_*_slice*_top_patches.png
        └── case_*_*_slice*_bottom_patches.png
```

## Output Files

### Always Generated

**results.txt**
- Test loss and accuracy
- Number of samples
- Checkpoint information

**predictions.csv**
- `case_id`: Case identifier
- `true_label`: Ground truth (0=benign, 1=high-grade)
- `predicted_label`: Model prediction
- `prob_benign`: Probability for benign class
- `prob_high_grade`: Probability for high-grade class
- `correct`: Boolean indicating correct prediction

**confusion_matrix.png**
- confusion matrix with counts for predictions on the test set

**data_splits.npz**
Contains case IDs for each split:
- `train_cases`: Training set case IDs
- `val_cases`: Validation set case IDs
- `test_cases`: Test set case IDs

### Generated Only During Training
- checkpoint files

### Optional: Attention Analysis (--analyze_attention)

**attention_summary.txt**
- Most attended stain per case
- Stain-level attention weights
- Slice-level attention patterns

**patch_attention/ folder**
- Top N most attended patches per slice (highest/lowest slice-level attention)
- Bottom N least attended patches per slice
- Images (after transformation) with attention weights

**case_effective_patches/ folder**
- Top N patches across entire case using effective attention (stain × slice × patch weights)
- Bottom N patches across entire case using effective attention
- Provides global view of most important patches per case

**plots/ folder**
- `effective_patch_attn_distro_case_*.png`: Per-case histograms of effective patch attention
- Shows distribution and concentration of attention within each case

**slice_attention/ folder**
- `slice_attn_rankplot_case_*.png`: Per-case slice attention rankings by stain
- Bar plots showing which slices get most attention within each stain
- Includes uniform attention reference line

**Analysis CSVs and Summaries**
- `top_effective_patches_per_case_5.0pct.csv`: Detailed data on top 5% patches per case
- `top_effective_patches_per_case_summary_5.0pct.txt`: Human-readable summary of top patches
- Includes stain distribution and slice coverage statistics

## Data Format

The model expects:
- **Patches**: PNG images organized by case and stain
- **Labels CSV**: Case IDs with corresponding class labels
- **Naming Convention**: `case_{case_id}_{slice_id}_{stain}_patch{n}.png`

## Project Structure

```
MIL_trainer_9Dec_JointFinal/
├── config.py              # Configuration and paths
├── data_utils.py          # Data loading and preprocessing
├── models.py              # Model architectures
├── dataset.py             # Dataset classes and transforms
├── trainer.py             # Training and validation logic
├── attention_analysis.py  # Attention visualization
├── utils.py               # Helper functions
├── main.py                # Main training script
├── requirements.txt       # Dependencies
├── data_splits/           # Pre-generated train/val/test splits
│   ├── data_splits_01.npz
│   ├── data_splits_02.npz
│   ├── data_splits_03.npz
│   ├── data_splits_04.npz
│   └── data_splits_05.npz
└── sbatch_files/          # SLURM job scripts
    ├── README.txt
    ├── run_training_1st_splits.sbatch
    ├── run_training_2nd_splits.sbatch
    ├── run_training_3rd_splits.sbatch
    ├── run_training_4th_splits.sbatch
    └── run_training_5th_splits.sbatch
```