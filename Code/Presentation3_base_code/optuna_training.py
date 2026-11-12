#!/usr/bin/env python3
"""
Optuna tuning for the Hierarchical Attention MIL model.

What this script does
---------------------
- Tunes 6 hyperparameters:
    1) weight_decay
    2) patch projector dropout
    3) classifier dropout
    4) initial learning rate
    5) benign-class weight in CrossEntropyLoss
    6) patch-attention entropy regularization lambda
- Uses 5 predefined train/val/test case splits from no_train_val_leakage_splits.
- For each Optuna trial, trains one model per split and averages validation loss
  across the 5 validation splits at each epoch.
- Uses a 16-point Sobol startup design, then switches to TPE suggestions.
- Uses MedianPruner with:
    * no pruning for first 6 trials
    * pruning warm-up of 6 epochs
- Runs at most 30 epochs per trial.

Objective returned to Optuna
----------------------------
For each epoch, the script computes the mean validation loss across the 5 splits.
The trial objective is the *best epoch-level mean validation loss*.
"""

import argparse
import copy
import glob
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quasirandom import SobolEngine
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DATA_PATHS, MODEL_CONFIG, TRAINING_CONFIG
from data_utils import (
    build_case_dict,
    build_slice_to_class_map,
    get_all_patch_files,
    group_patches_by_slice,
    load_labels,
)
from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
from utils import get_device, load_data_splits, print_model_summary, set_seed


EMB_DIR = "/projects/e32998/patches_varsize_pooled4096"


# -----------------------------------------------------------------------------
# Model with separately tunable patch-projector and classifier dropout
# -----------------------------------------------------------------------------
class GatedAttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        v = self.attention_V(x)
        u = self.attention_U(x)
        gated = v * u
        scores = self.attention_w(gated)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * x).sum(dim=1)
        if return_weights:
            return pooled, weights.squeeze(-1)
        return pooled


class TunableHierarchicalAttnMIL(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 512,
        patch_proj_dropout: float = 0.3,
        classifier_dropout: float = 0.3,
        pooled_dim: int = 4096,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pooled_dim = pooled_dim

        self.patch_projector = nn.Sequential(
            nn.Linear(self.pooled_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(patch_proj_dropout),
        )

        hidden_dim = MODEL_CONFIG["attention_hidden_dim"]
        self.patch_attention = GatedAttentionPool(embed_dim, hidden_dim)
        self.stain_attention = GatedAttentionPool(embed_dim, hidden_dim)
        self.case_attention = GatedAttentionPool(embed_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def process_single_stain(
        self,
        slice_list: List[torch.Tensor],
        stain_name: str,
        return_attn_weights: bool = False,
    ):
        device = next(self.parameters()).device
        slice_embeddings = []
        patch_weight_list = []

        for slice_tensor in slice_list:
            if slice_tensor.dim() != 2:
                raise ValueError(
                    f"[{stain_name}] Expected pooled features of shape (P, {self.pooled_dim}), got {tuple(slice_tensor.shape)}"
                )
            if slice_tensor.size(1) != self.pooled_dim:
                raise ValueError(
                    f"[{stain_name}] Expected pooled_dim={self.pooled_dim}, got {slice_tensor.size(1)}"
                )

            pooled = slice_tensor.to(device, non_blocking=True)
            patch_embeddings = self.patch_projector(pooled)

            if return_attn_weights:
                slice_emb, patch_weights = self.patch_attention(
                    patch_embeddings.unsqueeze(0), return_weights=True
                )
                patch_weight_list.append(patch_weights.squeeze(0))
            else:
                slice_emb = self.patch_attention(patch_embeddings.unsqueeze(0))

            slice_embeddings.append(slice_emb.squeeze(0))

        if not slice_embeddings:
            return None, None

        stain_slice_embeddings = torch.stack(slice_embeddings)

        if return_attn_weights:
            stain_emb, stain_weights = self.stain_attention(
                stain_slice_embeddings.unsqueeze(0), return_weights=True
            )
            attn_info = {
                "slice_weights": stain_weights.squeeze(0).detach(),
                "patch_weights": patch_weight_list,
            }
        else:
            stain_emb = self.stain_attention(stain_slice_embeddings.unsqueeze(0))
            attn_info = None

        return stain_emb.squeeze(0), attn_info

    def forward(
        self,
        stain_slices_dict: Dict[str, List[torch.Tensor]],
        return_attn_weights: bool = False,
    ):
        stain_embeddings = []
        stain_names = []
        stain_attention_weights: Dict[str, Any] = {}

        for stain_name, slice_list in stain_slices_dict.items():
            if not slice_list:
                continue

            stain_emb, stain_attn_info = self.process_single_stain(
                slice_list, stain_name, return_attn_weights
            )

            if stain_emb is not None:
                stain_embeddings.append(stain_emb)
                stain_names.append(stain_name)
                if return_attn_weights and stain_attn_info is not None:
                    stain_attention_weights[stain_name] = stain_attn_info

        if not stain_embeddings:
            logits = torch.zeros(self.num_classes, device=next(self.parameters()).device)
            if return_attn_weights:
                return logits, {}
            return logits

        case_stain_embeddings = torch.stack(stain_embeddings)

        if return_attn_weights:
            case_emb, case_weights = self.case_attention(
                case_stain_embeddings.unsqueeze(0), return_weights=True
            )
            all_weights = {
                "case_weights": case_weights.squeeze(0),
                "stain_weights": stain_attention_weights,
                "stain_order": stain_names,
            }
        else:
            case_emb = self.case_attention(case_stain_embeddings.unsqueeze(0))

        logits = self.classifier(case_emb.squeeze(0))
        if return_attn_weights:
            return logits, all_weights
        return logits


# -----------------------------------------------------------------------------
# Fold trainer
# -----------------------------------------------------------------------------
class FoldTrainer:
    def __init__(self, model: nn.Module, device: str, hparams: Dict[str, float], max_epochs: int):
        self.model = model.to(device)
        self.device = device
        self.hparams = hparams

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=TRAINING_CONFIG.get("scheduler_factor", 0.2139),
            patience=TRAINING_CONFIG.get("scheduler_patience", 4),
            min_lr=TRAINING_CONFIG.get("scheduler_min_lr", 1e-6),
        )
        class_weights = torch.tensor(
            [hparams["class_weight_benign"], 1.0],
            dtype=torch.float32,
            device=device,
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.entropy_lambda = hparams["entropy_lambda"]
        self.entropy_eps = TRAINING_CONFIG.get("patch_entropy_eps", 1e-8)
        self.normalize_patch_entropy = bool(TRAINING_CONFIG.get("normalize_patch_entropy", True))
        self.max_epochs = max_epochs

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def _forward_one_case(self, case_data: Dict[str, Any]):
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)
        logits = self.model(stain_slices)
        if logits.dim() != 1:
            raise ValueError(f"Expected logits shape (num_classes,), got {tuple(logits.shape)}")
        logits = logits.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        return logits, label

    def _patch_attention_entropy(self, attention_info: Dict[str, Any]) -> torch.Tensor:
        if not attention_info:
            return torch.tensor(0.0, device=self.device)

        entropies = []
        stain_dict = attention_info.get("stain_weights", {})
        for stain_info in stain_dict.values():
            for patch_weights in stain_info.get("patch_weights", []):
                if patch_weights is None or patch_weights.numel() == 0:
                    continue
                p = patch_weights.to(self.device).clamp_min(self.entropy_eps)
                entropy = -(p * torch.log(p)).sum()
                if self.normalize_patch_entropy and p.numel() > 1:
                    entropy = entropy / torch.log(torch.tensor(float(p.numel()), device=self.device))
                entropies.append(entropy)

        if not entropies:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(entropies).mean()

    def _forward_one_case_with_entropy(self, case_data: Dict[str, Any]):
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)
        logits, attention_info = self.model(stain_slices, return_attn_weights=True)
        if logits.dim() != 1:
            raise ValueError(f"Expected logits shape (num_classes,), got {tuple(logits.shape)}")
        logits = logits.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        entropy = self._patch_attention_entropy(attention_info)
        return logits, label, entropy

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        running_loss = 0.0
        n = 0

        for batch in train_loader:
            if not batch:
                continue
            case_data = batch[0]

            if self.entropy_lambda > 0.0:
                logits, label, entropy = self._forward_one_case_with_entropy(case_data)
                ce_loss = self.criterion(logits, label)
                loss = ce_loss - self.entropy_lambda * entropy
            else:
                logits, label = self._forward_one_case(case_data)
                loss = self.criterion(logits, label)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            n += 1

        avg_loss = running_loss / max(n, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        n = 0

        with torch.inference_mode():
            for batch in val_loader:
                if not batch:
                    continue
                case_data = batch[0]
                logits, label = self._forward_one_case(case_data)
                loss = self.criterion(logits, label)
                total_loss += float(loss.item())
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == label).sum().item())
                n += 1

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(acc)

        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_epoch = len(self.val_losses)

        return avg_loss, acc

    def step_scheduler(self, val_loss: float) -> None:
        self.scheduler.step(val_loss)


# -----------------------------------------------------------------------------
# Data preparation for all 5 split files
# -----------------------------------------------------------------------------
def prepare_fold_loaders(args: argparse.Namespace, device: str):
    print("=" * 80)
    print("PREPARING ALL 5 SPLITS FOR OPTUNA")
    print("=" * 80)
    labels = load_labels(args.labels_csv)
    all_files = get_all_patch_files(args.patches_dir)
    patches = group_patches_by_slice(all_files, args.patches_dir)
    slice_to_class = build_slice_to_class_map(patches, labels)

    split_paths = sorted(glob.glob(os.path.join(args.splits_dir, args.splits_pattern)))
    if len(split_paths) != 5:
        raise ValueError(
            f"Expected 5 split files in {args.splits_dir} matching {args.splits_pattern}, found {len(split_paths)}"
        )

    pin_memory = str(device).startswith("cuda")
    persistent_workers = args.num_workers > 0
    fold_data = []

    for fold_idx, split_path in enumerate(split_paths, start=1):
        split_data = load_data_splits(split_path)
        train_cases = set(split_data["train_cases"])
        val_cases = set(split_data["val_cases"])

        train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in train_cases]
        val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in val_cases]

        train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
        val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)

        train_ds = StainBagCasePooledFeatureDataset(
            train_case_dict,
            train_label_map,
            embeddings_dir=args.embeddings_dir,
            per_slice_cap=args.per_slice_cap,
            max_slices_per_stain=args.max_slices_per_stain,
            shuffle_patches=True,
        )
        val_ds = StainBagCasePooledFeatureDataset(
            val_case_dict,
            val_label_map,
            embeddings_dir=args.embeddings_dir,
            per_slice_cap=args.per_slice_cap,
            max_slices_per_stain=args.max_slices_per_stain,
            shuffle_patches=False,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=case_collate_fn,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=case_collate_fn,
            persistent_workers=persistent_workers,
        )

        print(
            f"Fold {fold_idx}: {os.path.basename(split_path)} | "
            f"Train cases={len(train_ds)} | Val cases={len(val_ds)} | "
            f"Train batches={len(train_loader)} | Val batches={len(val_loader)}"
        )

        fold_data.append(
            {
                "fold_idx": fold_idx,
                "split_path": split_path,
                "train_loader": train_loader,
                "val_loader": val_loader,
            }
        )

    return fold_data


# -----------------------------------------------------------------------------
# Hyperparameter space + Sobol startup points
# -----------------------------------------------------------------------------
SEARCH_SPACE = {
    "learning_rate": {"low": 3e-5, "high": 1e-3, "log": True},
    "weight_decay": {"low": 1e-6, "high": 2e-3, "log": True},
    "patch_proj_dropout": {"low": 0.15, "high": 0.55, "log": False},
    "classifier_dropout": {"low": 0.15, "high": 0.55, "log": False},
    "class_weight_benign": {"low": 2.0, "high": 4.0, "log": False},
    "entropy_lambda": {"low": 1e-5, "high": 1e-2, "log": True},
}

ORDERED_PARAMS = [
    "learning_rate",
    "weight_decay",
    "patch_proj_dropout",
    "classifier_dropout",
    "class_weight_benign",
    "entropy_lambda",
]


def map_unit_to_range(x: float, low: float, high: float, log_scale: bool) -> float:
    if log_scale:
        lo = math.log(low)
        hi = math.log(high)
        return float(math.exp(lo + x * (hi - lo)))
    return float(low + x * (high - low))


def generate_sobol_startup_trials(n_points: int, seed: int) -> List[Dict[str, float]]:
    engine = SobolEngine(dimension=len(ORDERED_PARAMS), scramble=True, seed=seed)
    # draw(n) works for arbitrary n; 16 requested by user
    pts = engine.draw(n_points).cpu().numpy()
    trials = []
    for row in pts:
        params = {}
        for i, name in enumerate(ORDERED_PARAMS):
            spec = SEARCH_SPACE[name]
            params[name] = map_unit_to_range(row[i], spec["low"], spec["high"], spec["log"])
        trials.append(params)
    return trials


# -----------------------------------------------------------------------------
# Objective
# -----------------------------------------------------------------------------
def suggest_hparams(trial: optuna.Trial) -> Dict[str, float]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 3e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 2e-3, log=True),
        "patch_proj_dropout": trial.suggest_float("patch_proj_dropout", 0.15, 0.55),
        "classifier_dropout": trial.suggest_float("classifier_dropout", 0.15, 0.55),
        "class_weight_benign": trial.suggest_float("class_weight_benign", 2.0, 4.0),
        "entropy_lambda": trial.suggest_float("entropy_lambda", 1e-5, 1e-2, log=True),
    }


def build_model_from_hparams(hparams: Dict[str, float]) -> nn.Module:
    return TunableHierarchicalAttnMIL(
        num_classes=MODEL_CONFIG["num_classes"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        patch_proj_dropout=hparams["patch_proj_dropout"],
        classifier_dropout=hparams["classifier_dropout"],
        pooled_dim=4096,
    )


def objective_factory(
    fold_data: List[Dict[str, Any]],
    device: str,
    max_epochs: int,
    base_seed: int,
):
    def objective(trial: optuna.Trial) -> float:
        hparams = suggest_hparams(trial)

        print("\n" + "=" * 80)
        print(f"Trial {trial.number}")
        print("=" * 80)
        for k, v in hparams.items():
            if isinstance(v, float) and v < 0.01:
                print(f"{k}: {v:.3e}")
            else:
                print(f"{k}: {v:.6f}")

        # trial-specific reproducibility
        set_seed(base_seed + trial.number)

        fold_trainers: List[FoldTrainer] = []
        for fold_info in fold_data:
            model = build_model_from_hparams(hparams)
            trainer = FoldTrainer(model=model, device=device, hparams=hparams, max_epochs=max_epochs)
            fold_trainers.append(trainer)

        best_avg_val_loss = float("inf")
        best_epoch = -1
        best_epoch_fold_val_losses: Optional[List[float]] = None
        best_epoch_fold_val_accs: Optional[List[float]] = None

        try:
            for epoch in range(max_epochs):
                epoch_train_losses = []
                epoch_val_losses = []
                epoch_val_accs = []

                print(f"\nEpoch {epoch + 1}/{max_epochs}")
                for fold_info, trainer in zip(fold_data, fold_trainers):
                    train_loss = trainer.train_epoch(fold_info["train_loader"])
                    val_loss, val_acc = trainer.validate(fold_info["val_loader"])
                    trainer.step_scheduler(val_loss)

                    epoch_train_losses.append(train_loss)
                    epoch_val_losses.append(val_loss)
                    epoch_val_accs.append(val_acc)

                    print(
                        f"  Fold {fold_info['fold_idx']}: "
                        f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
                    )

                mean_train_loss = float(np.mean(epoch_train_losses))
                mean_val_loss = float(np.mean(epoch_val_losses))
                mean_val_acc = float(np.mean(epoch_val_accs))

                print(
                    f"  --> Mean across 5 folds: "
                    f"train_loss={mean_train_loss:.4f} | val_loss={mean_val_loss:.4f} | val_acc={mean_val_acc:.4f}"
                )

                if mean_val_loss < best_avg_val_loss:
                    best_avg_val_loss = mean_val_loss
                    best_epoch = epoch + 1
                    best_epoch_fold_val_losses = list(epoch_val_losses)
                    best_epoch_fold_val_accs = list(epoch_val_accs)

                trial.report(mean_val_loss, step=epoch)
                if trial.should_prune():
                    trial.set_user_attr("pruned_epoch", epoch + 1)
                    raise optuna.TrialPruned()

            # store summary attrs
            trial.set_user_attr("best_epoch", best_epoch)
            trial.set_user_attr("best_avg_val_loss", best_avg_val_loss)
            trial.set_user_attr("best_epoch_fold_val_losses", best_epoch_fold_val_losses)
            trial.set_user_attr("best_epoch_fold_val_accs", best_epoch_fold_val_accs)
            trial.set_user_attr(
                "per_fold_best_val_loss",
                [float(tr.best_val_loss) for tr in fold_trainers],
            )

            return best_avg_val_loss
        finally:
            del fold_trainers
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


# -----------------------------------------------------------------------------
# Saving results
# -----------------------------------------------------------------------------
def save_study_outputs(study: optuna.Study, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    csv_path = os.path.join(output_dir, "optuna_trials.csv")
    df.to_csv(csv_path, index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    summary = {
        "study_name": study.study_name,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_trials_total": len(study.trials),
        "n_trials_complete": len(completed),
        "n_trials_pruned": sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
    }

    if completed:
        best = study.best_trial
        summary["best_trial_number"] = best.number
        summary["best_value"] = best.value
        summary["best_params"] = best.params
        summary["best_user_attrs"] = best.user_attrs

    with open(os.path.join(output_dir, "study_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # HTML plots
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, "optimization_history.html"))
    except Exception as exc:
        print(f"Could not write optimization history plot: {exc}")

    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, "param_importances.html"))
    except Exception as exc:
        print(f"Could not write parameter importance plot: {exc}")

    try:
        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, "parallel_coordinate.html"))
    except Exception as exc:
        print(f"Could not write parallel coordinate plot: {exc}")

    print(f"Saved Optuna outputs to: {output_dir}")
    print(f"Trials CSV: {csv_path}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold Optuna tuning for MIL")
    parser.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
    parser.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
    parser.add_argument("--embeddings_dir", type=str, default=EMB_DIR)
    parser.add_argument(
        "--splits_dir",
        type=str,
        default="/projects/e32998/STAT390_Krish/Code/Code9_no_leakage/no_train_val_leakage_splits",
    )
    parser.add_argument("--splits_pattern", type=str, default="data_splits_new_0*.npz")
    parser.add_argument("--output_dir", type=str, default="./optuna_5fold_results")
    parser.add_argument("--study_name", type=str, default="mil_5fold_sobol16_tpe64")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g. sqlite:///optuna_5fold.db")
    parser.add_argument("--n_trials", type=int, default=64)
    parser.add_argument("--sobol_startup_trials", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pruning_warmup_epochs", type=int, default=6)
    parser.add_argument("--no_prune_first_trials", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=TRAINING_CONFIG.get("random_state", 42))
    parser.add_argument("--per_slice_cap", type=int, default=500)
    parser.add_argument("--max_slices_per_stain", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.abspath(f"{args.output_dir}_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("OPTUNA 5-FOLD HYPERPARAMETER TUNING")
    print("=" * 80)
    print(vars(args))
    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")

    fold_data = prepare_fold_loaders(args, device)

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=0)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.no_prune_first_trials,
        n_warmup_steps=args.pruning_warmup_epochs,
    )

    if args.storage is None:
        storage = f"sqlite:///{os.path.join(args.output_dir, 'optuna_5fold.db')}"
    else:
        storage = args.storage

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,
    )

    # enqueue the 16 Sobol startup points only if the study is still empty
    if len(study.trials) == 0:
        sobol_trials = generate_sobol_startup_trials(args.sobol_startup_trials, args.seed)
        for params in sobol_trials:
            study.enqueue_trial(params)
        print(f"Enqueued {len(sobol_trials)} Sobol startup trials.")
    else:
        print(f"Study already has {len(study.trials)} trials; Sobol startup trials were not re-enqueued.")

    objective = objective_factory(
        fold_data=fold_data,
        device=device,
        max_epochs=args.epochs,
        base_seed=args.seed,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    save_study_outputs(study, args.output_dir)

    if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 0:
        print("\nBest trial summary")
        print("-" * 80)
        print(f"Best trial #: {study.best_trial.number}")
        print(f"Best objective (best epoch mean val loss across 5 folds): {study.best_value:.6f}")
        for k, v in study.best_params.items():
            if isinstance(v, float) and v < 0.01:
                print(f"  {k}: {v:.3e}")
            else:
                print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
