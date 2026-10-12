"""
Dataset classes for MIL training
"""
import random
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from typing import Dict, List, Any, Optional, Tuple
import torchvision.transforms as transforms

from config import MODEL_CONFIG, IMAGE_CONFIG


class StainBagCaseDataset(Dataset):
    """
    Dataset for multi-stain MIL training
    
    For each CASE returns:
    {
        "case_id": int/str,
        "stain_slices": {
            "h&e":   [Tensor(P1, C, H, W), Tensor(P2, C, H, W), ...],
            "melan": [...],
            "sox10": [...],
        },
        "label": LongTensor scalar
    }
    """
    
    def __init__(
        self,
        case_dict: Dict[Any, Dict[str, List[List[str]]]],
        label_map: Dict[Any, int],
        transform=None,
        stains: Tuple[str, ...] = None,
        per_slice_cap: Optional[int] = None,
        max_slices_per_stain: Optional[int] = None,
        shuffle_patches: bool = True,
        drop_empty_slices: bool = True,
    ):
        """
        Args:
            case_dict: {case_id: {stain: [[patches_of_slice1], [patches_of_slice2], ...]}}
            label_map: {case_id: int_label}
            transform: image transforms
            stains: tuple of stain names to process
            per_slice_cap: max patches per slice
            max_slices_per_stain: max slices per stain
            shuffle_patches: whether to shuffle patches within slices
            drop_empty_slices: whether to drop slices that fail to load any patch
        """
        self.transform = transform
        self.stains = list(stains) if stains else list(MODEL_CONFIG['stains'])
        self.per_slice_cap = per_slice_cap if per_slice_cap is not None else MODEL_CONFIG['per_slice_cap']
        self.max_slices_per_stain = max_slices_per_stain if max_slices_per_stain is not None else MODEL_CONFIG['max_slices_per_stain']
        self.shuffle_patches = shuffle_patches
        self.drop_empty_slices = drop_empty_slices
        
        # Flatten case_dict to an indexable list; normalize stain keys
        self.items = []
        for case_id, stain_map in case_dict.items():
            if case_id not in label_map:
                continue
            
            # Normalize stain keys to lowercase
            norm_map = {}
            for k, v in stain_map.items():
                kk = k.lower()
                norm_map[kk] = v
            
            self.items.append((case_id, norm_map))
        
        self.label_map = label_map
    
    def __len__(self):
        return len(self.items)
    
    def _load_slice_tensor(self, paths: List[str]) -> Optional[torch.Tensor]:
        """
        Load one slice from list of patch paths -> Tensor(P, C, H, W)
        Applies transform; shuffles & caps per-slice; skips unreadable images
        """
        patch_paths = list(paths)
        if self.shuffle_patches:
            random.shuffle(patch_paths)
        
        if self.per_slice_cap and len(patch_paths) > self.per_slice_cap:
            patch_paths = patch_paths[:self.per_slice_cap]
        
        imgs = []
        for p in patch_paths:
            try:
                img = Image.open(p).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            except (Exception, UnidentifiedImageError):
                # Skip unreadable images
                continue
        
        if len(imgs) == 0:
            return None
        return torch.stack(imgs)  # (P, C, H, W)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        case_id, stain_map = self.items[idx]
        label = torch.tensor(self.label_map[case_id], dtype=torch.long)
        
        stain_slices: Dict[str, List[torch.Tensor]] = {}
        for stain in self.stains:
            # Get list of slices (each slice is list[str] of patch paths)
            slice_lists = stain_map.get(stain, []) or []
            
            # Optionally cap the number of slices per stain
            if self.max_slices_per_stain is not None and len(slice_lists) > self.max_slices_per_stain:
                slice_lists = slice_lists[:self.max_slices_per_stain]
            
            tensors_for_stain: List[torch.Tensor] = []
            for sl in slice_lists:
                if not sl:
                    continue
                sl_tensor = self._load_slice_tensor(sl)
                if sl_tensor is None:
                    if self.drop_empty_slices:
                        continue
                    else:
                        # Could represent empty as zero-length tensor
                        continue
                tensors_for_stain.append(sl_tensor)
            
            stain_slices[stain] = tensors_for_stain
        
        return {
            "case_id": case_id,
            "stain_slices": stain_slices,  # dict[stain] -> list[Tensor(P,C,H,W)]
            "label": label,
        }


def case_collate_fn(batch):
    """
    Collate function for MIL dataset
    Keep variable-length structures; model.forward expects List[case_dict]
    Use batch_size=1 to avoid padding/masking
    """
    return batch


def create_transforms(is_training: bool = True) -> transforms.Compose:
    """
    Create image transforms for training or validation
    """
    if is_training:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_CONFIG['image_size'][0]),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_CONFIG['normalize_mean'],
                std=IMAGE_CONFIG['normalize_std']
            )
        ])
    else:
        # Validation/test transforms without augmentation
        transform = transforms.Compose([
            transforms.Resize(IMAGE_CONFIG['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGE_CONFIG['normalize_mean'],
                std=IMAGE_CONFIG['normalize_std']
            )
        ])
    
    return transform