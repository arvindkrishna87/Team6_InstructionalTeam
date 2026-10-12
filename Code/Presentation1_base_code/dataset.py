"""
Dataset classes for MIL training
"""
import os
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Tuple

from config import MODEL_CONFIG


class StainBagCasePooledFeatureDataset(Dataset):
    """
    Like StainBagCaseDataset, but loads precomputed pooled feature vectors (.pt)
    instead of loading images.

    For each CASE returns:
    {
        "case_id": int/str,
        "stain_slices": {
            "h&e":   [Tensor(P1, F4), Tensor(P2, F4), ...],
            ...
        },
        "label": LongTensor scalar
    }
    where F4 = 4*F (typically 4096 for DenseNet121 with AdaptiveAvgPool2d((2,2))).
    """

    def __init__(
        self,
        case_dict: Dict[Any, Dict[str, List[List[str]]]],
        label_map: Dict[Any, int],
        embeddings_dir: str,
        stains: Tuple[str, ...] = None,
        per_slice_cap: Optional[int] = None,
        max_slices_per_stain: Optional[int] = None,
        shuffle_patches: bool = True,
        drop_empty_slices: bool = True,
        # NEW: force a consistent dtype so fp16 embeddings won't crash fp32 model
        feature_dtype: torch.dtype = torch.float32,
    ):
        self.embeddings_dir = embeddings_dir
        self.stains = list(stains) if stains else list(MODEL_CONFIG["stains"])
        self.per_slice_cap = per_slice_cap if per_slice_cap is not None else MODEL_CONFIG["per_slice_cap"]
        self.max_slices_per_stain = (
            max_slices_per_stain if max_slices_per_stain is not None else MODEL_CONFIG["max_slices_per_stain"]
        )
        self.shuffle_patches = shuffle_patches
        self.drop_empty_slices = drop_empty_slices

        # NEW
        self.feature_dtype = feature_dtype

        self.items = []
        for case_id, stain_map in case_dict.items():
            if case_id not in label_map:
                continue
            norm_map = {k.lower(): v for k, v in stain_map.items()}
            self.items.append((case_id, norm_map))
        self.label_map = label_map

    def __len__(self):
        return len(self.items)

    def _patch_path_to_feat_path(self, patch_path: str) -> str:
        # patch_path is full path like /projects/.../patches/<fname>.png
        fname = os.path.basename(patch_path)
        feat_name = os.path.splitext(fname)[0] + ".pt"
        return os.path.join(self.embeddings_dir, feat_name)

    def _load_slice_tensor(self, patch_paths: List[str]) -> Optional[torch.Tensor]:
        patch_paths = list(patch_paths)
        if self.shuffle_patches:
            random.shuffle(patch_paths)

        if self.per_slice_cap and len(patch_paths) > self.per_slice_cap:
            patch_paths = patch_paths[:self.per_slice_cap]

        vecs = []
        for p in patch_paths:
            fp = self._patch_path_to_feat_path(p)
            try:
                v = torch.load(fp, map_location="cpu")  # (4096,)
                if v is None:
                    continue

                # Ensure 1D
                if v.dim() != 1:
                    v = v.view(-1)

                # CRITICAL FIX: cast to float32 (or chosen dtype) so Half embeddings won't crash Float model
                if v.dtype != self.feature_dtype:
                    v = v.to(self.feature_dtype)

                vecs.append(v)
            except Exception:
                continue

        if len(vecs) == 0:
            return None

        return torch.stack(vecs, dim=0)  # (P, 4096)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, stain_map = self.items[idx]
        label = torch.tensor(self.label_map[case_id], dtype=torch.long)

        stain_slices: Dict[str, List[torch.Tensor]] = {}
        for stain in self.stains:
            slice_lists = stain_map.get(stain, []) or []

            if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                slice_lists = slice_lists[:self.max_slices_per_stain]

            tensors_for_stain: List[torch.Tensor] = []
            for sl in slice_lists:
                if not sl:
                    continue
                sl_tensor = self._load_slice_tensor(sl)

                if sl_tensor is None:
                    if self.drop_empty_slices:
                        # skip empty slice instead of crashing
                        continue
                    raise RuntimeError(f"Empty slice encountered in case {case_id}, stain {stain}")

                tensors_for_stain.append(sl_tensor)

            stain_slices[stain] = tensors_for_stain

        return {"case_id": case_id, "stain_slices": stain_slices, "label": label}


def case_collate_fn(batch):
    """
    Collate function for MIL dataset
    Keep variable-length structures; model.forward expects List[case_dict]
    Use batch_size=1 to avoid padding/masking
    """
    return batch