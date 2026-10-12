"""
trainer.py

Training / validation / evaluation logic for the MIL model.

Adds optional patch-attention entropy regularization.

Assumptions (matching the pipeline):
- DataLoader returns a "batch" that is a list with one dict (because batch_size=1 and case_collate_fn returns batch)
  So we access case_data = batch[0].
- model(stain_slices) returns logits of shape (num_classes,) (no batch dim).
- label is a scalar LongTensor (or shape (1,)).

IMPORTANT for entropy regularization:
- The model must return patch attention weights WITHOUT .detach() when
  return_attn_weights=True, otherwise the entropy term will not backpropagate.

Entropy regularization options:
- mode="min":  loss = CE + lambda * entropy
               encourages lower-entropy / peakier patch attention
- mode="max":  loss = CE - lambda * entropy
               encourages higher-entropy / smoother patch attention
"""

import time
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import TRAINING_CONFIG, DEVICE


class MILTrainer:
    """
    Trainer for MIL model.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        self.model = model
        self.device = device if device is not None else DEVICE
        self.checkpoint_dir = checkpoint_dir

        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"],
        )

        # Loss with class weights
        class_weights = torch.tensor(
            TRAINING_CONFIG["class_weights"],
            dtype=torch.float32,
            device=self.device,
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Entropy regularization config
        self.use_patch_entropy_regularization = TRAINING_CONFIG.get(
            "use_patch_entropy_regularization", False
        )
        self.patch_entropy_lambda = float(
            TRAINING_CONFIG.get("patch_entropy_lambda", 0.0)
        )
        self.patch_entropy_mode = TRAINING_CONFIG.get(
            "patch_entropy_mode", "max"
        ).lower()

        if self.patch_entropy_mode not in {"min", "max"}:
            raise ValueError(
                f"patch_entropy_mode must be 'min' or 'max', got {self.patch_entropy_mode}"
            )

        self.entropy_eps = float(TRAINING_CONFIG.get("patch_entropy_eps", 1e-8))
        self.normalize_patch_entropy = bool(
            TRAINING_CONFIG.get("normalize_patch_entropy", False)
        )

        # LR scheduler
        self.scheduler = None
        if TRAINING_CONFIG.get("use_scheduler", False):
            scheduler_type = TRAINING_CONFIG.get("scheduler_type", "reduce_on_plateau")
            if scheduler_type == "reduce_on_plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    factor=TRAINING_CONFIG.get("scheduler_factor", 0.5),
                    patience=TRAINING_CONFIG.get("scheduler_patience", 3),
                    min_lr=TRAINING_CONFIG.get("scheduler_min_lr", 1e-6),
                )
            elif scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=TRAINING_CONFIG["epochs"],
                    eta_min=TRAINING_CONFIG.get("scheduler_min_lr", 1e-6),
                )
            else:
                raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

        # Early stopping
        self.use_early_stopping = TRAINING_CONFIG.get("early_stopping", False)
        self.early_stopping_patience = TRAINING_CONFIG.get("early_stopping_patience", 7)
        self.early_stopping_min_delta = TRAINING_CONFIG.get("early_stopping_min_delta", 0.001)
        self.early_stopping_min_epochs = TRAINING_CONFIG.get("early_stopping_min_epochs", 0)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # History
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.train_ce_losses = []
        self.train_patch_entropy_losses = []

    # ----------------------------
    # Core helpers
    # ----------------------------
    def _forward_one_case(self, case_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (logits_with_batch, label_with_batch)
        logits_with_batch: shape (1, num_classes)
        label_with_batch:  shape (1,)
        """
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)

        logits = self.model(stain_slices)  # (num_classes,)
        if logits.dim() != 1:
            raise ValueError(f"Expected model to return (num_classes,), got {tuple(logits.shape)}")

        logits = logits.unsqueeze(0)  # (1, num_classes)

        if label.dim() == 0:
            label = label.unsqueeze(0)  # (1,)

        return logits, label

    def _ensure_dir(self, path: Optional[str]) -> str:
        if path is None:
            raise ValueError("output_dir/checkpoint_dir must be provided (got None).")
        os.makedirs(path, exist_ok=True)
        return path

    def _compute_patch_attention_entropy(
        self,
        attention_info: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute mean entropy over all patch-attention vectors across all stains and slices.

        Expected structure:
        attention_info = {
            "case_weights": ...,
            "stain_weights": {
                stain_name: {
                    "slice_weights": Tensor[num_slices],
                    "patch_weights": List[Tensor[num_patches_in_slice]],
                },
                ...
            },
            "stain_order": ...
        }
        """
        if not attention_info:
            return torch.tensor(0.0, device=self.device)

        stain_weights_dict = attention_info.get("stain_weights", {})
        entropies = []

        for stain_dict in stain_weights_dict.values():
            patch_weight_list = stain_dict.get("patch_weights", [])
            for patch_w in patch_weight_list:
                if patch_w is None or patch_w.numel() == 0:
                    continue

                p = patch_w.to(self.device).clamp_min(self.entropy_eps)
                entropy = -(p * torch.log(p)).sum()

                if self.normalize_patch_entropy and p.numel() > 1:
                    entropy = entropy / torch.log(
                        torch.tensor(float(p.numel()), device=self.device)
                    )

                entropies.append(entropy)

        if len(entropies) == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(entropies).mean()

    def _forward_one_case_with_patch_entropy(
        self,
        case_data: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits_with_batch: (1, num_classes)
            label_with_batch:  (1,)
            patch_entropy:     scalar tensor
        """
        stain_slices = case_data["stain_slices"]
        label = case_data["label"].to(self.device)

        logits, attention_info = self.model(
            stain_slices, return_attn_weights=True
        )  # logits: (num_classes,)

        if logits.dim() != 1:
            raise ValueError(f"Expected model to return (num_classes,), got {tuple(logits.shape)}")

        logits = logits.unsqueeze(0)  # (1, num_classes)

        if label.dim() == 0:
            label = label.unsqueeze(0)  # (1,)

        patch_entropy = self._compute_patch_attention_entropy(attention_info)
        return logits, label, patch_entropy

    # ----------------------------
    # Train / Validate
    # ----------------------------
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        Returns average total training loss.
        """
        self.model.train()

        if len(train_loader) == 0:
            print("[WARN] train_loader is empty. Returning loss=0.0")
            self.train_losses.append(0.0)
            self.train_ce_losses.append(0.0)
            self.train_patch_entropy_losses.append(0.0)
            return 0.0

        running_loss = 0.0
        running_ce_loss = 0.0
        running_patch_entropy = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            if not batch:
                continue
            case_data = batch[0]

            if self.use_patch_entropy_regularization and self.patch_entropy_lambda > 0.0:
                logits, label, patch_entropy = self._forward_one_case_with_patch_entropy(case_data)
                ce_loss = self.criterion(logits, label)

                if self.patch_entropy_mode == "min":
                    loss = ce_loss + self.patch_entropy_lambda * patch_entropy
                else:  # "max"
                    loss = ce_loss - self.patch_entropy_lambda * patch_entropy

                running_patch_entropy += float(patch_entropy.item())
                running_ce_loss += float(ce_loss.item())
            else:
                logits, label = self._forward_one_case(case_data)
                ce_loss = self.criterion(logits, label)
                loss = ce_loss
                running_ce_loss += float(ce_loss.item())

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_loss = running_loss / max(num_batches, 1)
        avg_ce_loss = running_ce_loss / max(num_batches, 1)
        avg_patch_entropy = running_patch_entropy / max(num_batches, 1)

        self.train_losses.append(avg_loss)
        self.train_ce_losses.append(avg_ce_loss)
        self.train_patch_entropy_losses.append(avg_patch_entropy)

        return avg_loss

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        Returns (avg_loss, accuracy).

        Validation uses classification loss only.
        """
        self.model.eval()

        if len(val_loader) == 0:
            print("[WARN] val_loader is empty. Returning loss=0.0, acc=0.0")
            self.val_losses.append(0.0)
            self.val_accuracies.append(0.0)
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0
        n = 0

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                if not batch:
                    continue
                case_data = batch[0]

                logits, label = self._forward_one_case(case_data)
                loss = self.criterion(logits, label)
                total_loss += float(loss.item())

                pred = torch.argmax(logits, dim=1)  # (1,)
                correct += int((pred == label).sum().item())
                n += 1

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(acc)
        return avg_loss, acc

    # ----------------------------
    # Checkpointing
    # ----------------------------
    def save_checkpoint(
        self,
        epoch: int,
        arch: str = "HierarchicalAttnMIL",
        checkpoint_dir: Optional[str] = None,
        is_best: bool = False,
    ) -> str:
        """
        Save checkpoint and return filename.
        """
        ckpt_dir = checkpoint_dir if checkpoint_dir is not None else self.checkpoint_dir
        ckpt_dir = self._ensure_dir(ckpt_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(ckpt_dir, f"{timestamp}_{arch}_epoch{epoch}.pth")

        checkpoint = {
            "arch": arch,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "train_ce_losses": self.train_ce_losses,
            "train_patch_entropy_losses": self.train_patch_entropy_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "learning_rates": self.learning_rates,
            "best_val_loss": self.best_val_loss,
            "config": TRAINING_CONFIG,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

        if is_best:
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint updated: {best_path}")

        return filename

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load model checkpoint.
        Returns: epoch number loaded (int).
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint.get("train_losses", [])
        self.train_ce_losses = checkpoint.get("train_ce_losses", [])
        self.train_patch_entropy_losses = checkpoint.get("train_patch_entropy_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])
        self.learning_rates = checkpoint.get("learning_rates", [])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))

        epoch = int(checkpoint.get("epoch", 0))
        print(f"Checkpoint loaded: {checkpoint_path} (epoch={epoch})")
        return epoch

    # ----------------------------
    # Full training loop
    # ----------------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        start_epoch: int = 0,
        save_every: int = 1,
        arch: str = "HierarchicalAttnMIL",
    ):
        """
        Full training loop with scheduler + early stopping.
        """
        if epochs is None:
            epochs = TRAINING_CONFIG["epochs"]

        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.scheduler is not None:
            print(f"LR scheduler: {type(self.scheduler).__name__}")
        if self.use_patch_entropy_regularization and self.patch_entropy_lambda > 0.0:
            direction = "minimize entropy (peakier attention)" if self.patch_entropy_mode == "min" else "maximize entropy (smoother attention)"
            print(
                "Patch entropy regularization enabled "
                f"(lambda={self.patch_entropy_lambda}, mode={self.patch_entropy_mode}, "
                f"normalize={self.normalize_patch_entropy}) -> {direction}"
            )
        if self.use_early_stopping:
            print(
                "Early stopping enabled "
                f"(patience={self.early_stopping_patience}, "
                f"min_delta={self.early_stopping_min_delta}, "
                f"min_epochs={self.early_stopping_min_epochs})"
            )

        for epoch in range(start_epoch, epochs):
            current_lr = float(self.optimizer.param_groups[0]["lr"])
            print(f"\nEpoch {epoch + 1}/{epochs} (LR: {current_lr:.2e})")
            epoch_start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Record LR used this epoch
            self.learning_rates.append(current_lr)

            train_ce = self.train_ce_losses[-1] if self.train_ce_losses else train_loss
            train_ent = self.train_patch_entropy_losses[-1] if self.train_patch_entropy_losses else 0.0

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train CE: {train_ce:.4f} | "
                f"Train Patch Entropy: {train_ent:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            epoch_time = time.time() - epoch_start_time
            print(f"Epoch Time: {epoch_time:.2f}s")

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Check best / early stop
            is_best = False
            if val_loss < (self.best_val_loss - self.early_stopping_min_delta):
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                is_best = True
                print(f"New best validation loss: {val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                print(f"No improvement for {self.epochs_without_improvement} epoch(s)")

            # Save checkpoint periodically
            if save_every > 0 and ((epoch + 1) % save_every == 0):
                self.save_checkpoint(epoch + 1, arch=arch, is_best=is_best)

            # Early stopping
            if self.use_early_stopping:
                if (epoch + 1) >= self.early_stopping_min_epochs and self.epochs_without_improvement >= self.early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}. Best val loss: {self.best_val_loss:.4f}")
                    break

        print("\nTraining completed!")

    # ----------------------------
    # Evaluate + Outputs
    # ----------------------------
    def evaluate(
        self,
        test_loader: DataLoader,
        save_predictions: bool = True,
        output_dir: Optional[str] = None,
        checkpoint_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        Returns dict with metrics + predictions.
        """
        self.model.eval()

        if len(test_loader) == 0:
            print("[WARN] test_loader is empty. Returning zeros.")
            results = {
                "test_loss": 0.0,
                "test_accuracy": 0.0,
                "predictions": [],
                "true_labels": [],
                "case_ids": [],
                "prediction_probs": [],
                "num_samples": 0,
            }
            return results

        if save_predictions:
            out_dir = self._ensure_dir(output_dir)
        else:
            out_dir = output_dir

        total_loss = 0.0
        correct = 0
        n = 0

        predictions = []
        true_labels = []
        case_ids = []
        prediction_probs = []

        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Evaluating"):
                if not batch:
                    continue
                case_data = batch[0]
                case_id = case_data.get("case_id", None)

                logits, label = self._forward_one_case(case_data)
                loss = self.criterion(logits, label)
                total_loss += float(loss.item())

                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1)

                correct += int((pred == label).sum().item())
                n += 1

                case_ids.append(case_id)
                predictions.append(int(pred.cpu().item()))
                true_labels.append(int(label.cpu().item()))
                prediction_probs.append(probs.cpu().numpy()[0])

        avg_loss = total_loss / max(n, 1)
        acc = correct / max(n, 1)

        results = {
            "test_loss": avg_loss,
            "test_accuracy": acc,
            "predictions": predictions,
            "true_labels": true_labels,
            "case_ids": case_ids,
            "prediction_probs": prediction_probs,
            "num_samples": n,
        }

        print("\nTest Results:")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Acc:  {acc:.4f}")
        print(f"  Samples:   {n}")

        if save_predictions:
            self._save_predictions_csv(results, out_dir, checkpoint_name)
            self._save_confusion_matrix(results, out_dir)

        return results

    def _save_predictions_csv(
        self,
        results: Dict[str, Any],
        output_dir: str,
        checkpoint_name: Optional[str] = None,
    ) -> str:
        """
        Save per-case predictions to CSV in output_dir.
        """
        import pandas as pd

        self._ensure_dir(output_dir)

        csv_filename = "predictions.csv" if checkpoint_name is None else f"predictions_{os.path.basename(checkpoint_name)}.csv"
        csv_filename = csv_filename.replace(".pth", "").replace(" ", "_")
        csv_path = os.path.join(output_dir, csv_filename)

        probs_arr = results["prediction_probs"]
        if len(probs_arr) > 0:
            num_classes = len(probs_arr[0])
        else:
            num_classes = 0

        df_data = {
            "case_id": results["case_ids"],
            "true_label": results["true_labels"],
            "predicted_label": results["predictions"],
            "correct": [t == p for t, p in zip(results["true_labels"], results["predictions"])],
        }

        for c in range(num_classes):
            df_data[f"prob_class{c}"] = [float(p[c]) for p in probs_arr]

        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)

        print(f"Predictions saved to: {csv_path}")
        return csv_path

    def _save_confusion_matrix(self, results: Dict[str, Any], output_dir: str) -> str:
        """
        Save confusion matrix as PNG using seaborn.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        self._ensure_dir(output_dir)

        y_true = results["true_labels"]
        y_pred = results["predictions"]

        if len(y_true) == 0:
            print("[WARN] No samples; skipping confusion matrix.")
            return ""

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix - Test Set")

        cm_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(cm_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Confusion matrix saved to: {cm_path}")
        return cm_path


def count_patches_by_class(case_dict: Dict, label_map: Dict, split_name: str):
    """
    Count patches by class for analysis (based on patch path lists in case_dict).
    """
    from collections import defaultdict

    class_patch_counts = defaultdict(int)

    for case_id, stains in case_dict.items():
        if case_id not in label_map:
            continue

        label = label_map[case_id]
        total_patches = 0

        for stain_data in stains.values():
            for slice_patches in stain_data:
                total_patches += len(slice_patches)

        class_patch_counts[label] += total_patches

    print(f"\nPatch count by class for {split_name}:")
    print(f"  Benign (0):     {class_patch_counts[0]} patches")
    print(f"  High-grade (1): {class_patch_counts[1]} patches")

    return class_patch_counts