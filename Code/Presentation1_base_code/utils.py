"""
Utility functions for MIL training
"""
import os
import random
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, List, Any

from config import TRAINING_CONFIG


def set_seed(seed: int = None):
    """Set random seeds for reproducibility"""
    if seed is None:
        seed = TRAINING_CONFIG['random_state']
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed}")


def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def print_data_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Print comprehensive data summary"""
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print("\nData Summary:")
    print("=" * 50)
    
    # Label distribution per split
    print("\nLabel distribution per split:")
    label_dist = all_df.groupby(["split", "label"])["case_id"].nunique().unstack(fill_value=0)
    print(label_dist)
    
    # Patch statistics per stain per split
    stain_patch_cols = ["h&e_patches", "melan_patches", "sox10_patches"]
    print("\nMean patches per stain per split:")
    print(all_df.groupby("split")[stain_patch_cols].mean().round(1))
    
    print("\nMedian patches per stain per split:")
    print(all_df.groupby("split")[stain_patch_cols].median().round(1))
    
    # Missing stain analysis
    missing_cols = ["h&e_missing", "melan_missing", "sox10_missing"]
    print("\nMissing stain proportion:")
    print(all_df.groupby("split")[missing_cols].mean().round(3))
    
    # Total statistics
    print(f"\nTotal cases: {len(all_df)}")
    print(f"Total patches: {all_df['total_patches'].sum():,}")
    
    print("=" * 50)


def create_run_directory(base_dir: str = None) -> str:
    """Create a unique run directory with timestamp"""
    from datetime import datetime
    from config import DATA_PATHS
    
    if base_dir is None:
        base_dir = DATA_PATHS['runs_dir']
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Created run directory: {run_dir}")
    return run_dir


def save_data_splits(train_cases: List, val_cases: List, test_cases: List, 
                    save_dir: str = "./", name: str = "data_splits.npz"):
    """
    Save data splits for reproducibility
    
    Saves a .npz file containing the exact case IDs used in each split.
    This allows you to:
    - Reproduce the exact same train/val/test split
    - Verify no data leakage between splits
    - Compare results across different model runs
    - Analyze which cases were in which split
    
    The file contains three arrays:
    - train_cases: List of case IDs in training set
    - val_cases: List of case IDs in validation set
    - test_cases: List of case IDs in test set
    """
    splits = {
        'train_cases': train_cases,
        'val_cases': val_cases,
        'test_cases': test_cases
    }
    
    save_path = os.path.join(save_dir, name)
    np.savez(save_path, **splits)
    print(f"Data splits saved to: {save_path}")


def load_data_splits(load_path: str = "./data_splits.npz") -> Dict[str, List]:
    """Load previously saved data splits"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Data splits file not found: {load_path}")
    
    splits = np.load(load_path, allow_pickle=True)
    result = {
        'train_cases': splits['train_cases'].tolist(),
        'val_cases': splits['val_cases'].tolist(),
        'test_cases': splits['test_cases'].tolist()
    }
    
    print(f"Data splits loaded from: {load_path}")
    return result


def print_model_summary(model: torch.nn.Module):
    """Print model architecture summary"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Summary:")
    print("=" * 50)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50)


def analyze_attention_weights(attention_weights: Dict[str, Any], case_id: Any):
    """Analyze and print attention weights for a case"""
    print(f"\nAttention Analysis for Case {case_id}:")
    print("-" * 40)
    
    if 'case_weights' in attention_weights:
        case_weights = attention_weights['case_weights']
        stain_order = attention_weights.get('stain_order', [])
        
        print("Case-level attention (across stains):")
        for i, (stain, weight) in enumerate(zip(stain_order, case_weights)):
            print(f"  {stain}: {weight:.4f}")
    
    if 'stain_weights' in attention_weights:
        print("\nStain-level attention (across slices):")
        for stain, weights_dict in attention_weights['stain_weights'].items():
            slice_weights = weights_dict.get('slice_weights', [])
            print(f"  {stain}:")
            for i, weight in enumerate(slice_weights):
                print(f"    Slice {i+1}: {weight:.4f}")


def check_data_integrity(case_dict: Dict, label_map: Dict, split_name: str):
    """Check data integrity and report issues"""
    print(f"\nData Integrity Check for {split_name}:")
    
    issues = []
    
    # Check for cases without labels
    cases_without_labels = [case_id for case_id in case_dict.keys() if case_id not in label_map]
    if cases_without_labels:
        issues.append(f"Cases without labels: {len(cases_without_labels)}")
    
    # Check for empty cases
    empty_cases = []
    for case_id, stains in case_dict.items():
        total_patches = sum(len(slice_patches) for stain_data in stains.values() 
                          for slice_patches in stain_data)
        if total_patches == 0:
            empty_cases.append(case_id)
    
    if empty_cases:
        issues.append(f"Empty cases (no patches): {len(empty_cases)}")
    
    # Check for missing stains
    stain_coverage = defaultdict(int)
    for case_id, stains in case_dict.items():
        for stain in ['h&e', 'melan', 'sox10']:
            if stain in stains and len(stains[stain]) > 0:
                stain_coverage[stain] += 1
    
    total_cases = len(case_dict)
    print(f"Stain coverage:")
    for stain, count in stain_coverage.items():
        coverage = count / total_cases * 100 if total_cases > 0 else 0
        print(f"  {stain}: {count}/{total_cases} ({coverage:.1f}%)")
    
    if issues:
        print(f"Issues found: {', '.join(issues)}")
    else:
        print("No issues found")


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    else:
        import psutil
        memory = psutil.virtual_memory()
        used = memory.used / 1024**3  # GB
        total = memory.total / 1024**3  # GB
        return f"RAM - Used: {used:.2f}GB / {total:.2f}GB ({memory.percent:.1f}%)"