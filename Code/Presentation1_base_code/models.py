import torch
import torch.nn as nn
from typing import Dict, List

from config import MODEL_CONFIG


class GatedAttentionPool(nn.Module):
    """
    Gated attention pooling mechanism for MIL.

    a_i = w^T [tanh(Vh_i) * sigmoid(Uh_i)]
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        x: (B, M, D)
        """

        v = self.attention_V(x)     # (B, M, H)
        u = self.attention_U(x)     # (B, M, H)

        gated = v * u               # (B, M, H)

        scores = self.attention_w(gated)        # (B, M, 1)
        weights = torch.softmax(scores, dim=1)  # (B, M, 1)

        weighted_x = (weights * x).sum(dim=1)   # (B, D)

        if return_weights:
            return weighted_x, weights.squeeze(-1)

        return weighted_x


class HierarchicalAttnMIL(nn.Module):
    """
    Hierarchical Attention MIL model for multi-stain pathology images
    ASSUMES precomputed pooled features per patch: (P, 4096).
    """

    def __init__(
        self,
        num_classes: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.3,
        pooled_dim: int = 4096,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.pooled_dim = pooled_dim

        # Patch embedding
        self.patch_projector = nn.Sequential(
            nn.Linear(self.pooled_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention modules (NO dropout)
        self.patch_attention = GatedAttentionPool(
            embed_dim, MODEL_CONFIG["attention_hidden_dim"]
        )

        self.stain_attention = GatedAttentionPool(
            embed_dim, MODEL_CONFIG["attention_hidden_dim"]
        )

        self.case_attention = GatedAttentionPool(
            embed_dim, MODEL_CONFIG["attention_hidden_dim"]
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def process_single_stain(
        self,
        slice_list: List[torch.Tensor],
        stain_name: str,
        return_attn_weights: bool = False,
    ):

        slice_embeddings = []
        slice_attention_weights = []

        device = next(self.parameters()).device

        for slice_tensor in slice_list:

            if slice_tensor.dim() != 2:
                raise ValueError(
                    f"[{stain_name}] Expected pooled features with shape (P, {self.pooled_dim}), "
                    f"but got {tuple(slice_tensor.shape)}"
                )

            if slice_tensor.size(1) != self.pooled_dim:
                raise ValueError(
                    f"[{stain_name}] Expected pooled_dim={self.pooled_dim}, "
                    f"but got {slice_tensor.size(1)}"
                )

            pooled = slice_tensor.to(device, non_blocking=True)

            patch_embeddings = self.patch_projector(pooled)  # (P, D)

            if return_attn_weights:

                slice_emb, patch_weights = self.patch_attention(
                    patch_embeddings.unsqueeze(0),
                    return_weights=True
                )

                slice_attention_weights.append(
                    patch_weights.squeeze(0)
                )

            else:

                slice_emb = self.patch_attention(
                    patch_embeddings.unsqueeze(0)
                )

            slice_embeddings.append(slice_emb.squeeze(0))

        if not slice_embeddings:
            return None, None

        stain_slice_embeddings = torch.stack(slice_embeddings)

        if return_attn_weights:

            stain_emb, stain_weights = self.stain_attention(
                stain_slice_embeddings.unsqueeze(0),
                return_weights=True
            )

            stain_attention_info = {
                "slice_weights": stain_weights.squeeze(0).detach(),
                "patch_weights": slice_attention_weights,
            }

        else:

            stain_emb = self.stain_attention(
                stain_slice_embeddings.unsqueeze(0)
            )

            stain_attention_info = None

        return stain_emb.squeeze(0), stain_attention_info

    def forward(
        self,
        stain_slices_dict: Dict[str, List[torch.Tensor]],
        return_attn_weights: bool = False,
    ):

        stain_embeddings = []
        stain_names = []
        stain_attention_weights = {}

        for stain_name, slice_list in stain_slices_dict.items():

            if not slice_list:
                continue

            stain_emb, stain_attn_info = self.process_single_stain(
                slice_list,
                stain_name,
                return_attn_weights
            )

            if stain_emb is not None:

                stain_embeddings.append(stain_emb)
                stain_names.append(stain_name)

                if return_attn_weights and stain_attn_info is not None:
                    stain_attention_weights[stain_name] = stain_attn_info

        if not stain_embeddings:

            logits = torch.zeros(
                self.num_classes,
                device=next(self.parameters()).device
            )

            if return_attn_weights:
                return logits, {}

            return logits

        case_stain_embeddings = torch.stack(stain_embeddings)

        if return_attn_weights:

            case_emb, case_weights = self.case_attention(
                case_stain_embeddings.unsqueeze(0),
                return_weights=True
            )

            all_weights = {
                "case_weights": case_weights.squeeze(0),
                "stain_weights": stain_attention_weights,
                "stain_order": stain_names,
            }

        else:

            case_emb = self.case_attention(
                case_stain_embeddings.unsqueeze(0)
            )

        logits = self.classifier(case_emb.squeeze(0))

        if return_attn_weights:
            return logits, all_weights

        return logits


def create_model(
    num_classes: int = None,
    embed_dim: int = None,
    dropout: float = None,
    pooled_dim: int = 4096,
) -> HierarchicalAttnMIL:

    if num_classes is None:
        num_classes = MODEL_CONFIG["num_classes"]

    if embed_dim is None:
        embed_dim = MODEL_CONFIG["embed_dim"]

    if dropout is None:
        from config import TRAINING_CONFIG
        dropout = TRAINING_CONFIG.get("dropout", 0.3)

    return HierarchicalAttnMIL(
        num_classes=num_classes,
        embed_dim=embed_dim,
        dropout=dropout,
        pooled_dim=pooled_dim,
    )