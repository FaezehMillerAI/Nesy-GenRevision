from __future__ import annotations


try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - optional dependency guard
    torch = None
    nn = object
    F = None


class VisionEntityCrossAttention(nn.Module):
    """Scaled cosine attention between image patches and entity embeddings."""

    def __init__(self, vision_dim: int = 768, entity_dim: int = 256, hidden_dim: int = 256, tau: float = 0.07):
        if torch is None:
            raise ImportError("Install nesy-gen[torch] to use VisionEntityCrossAttention.")
        super().__init__()
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
        self.tau = tau

    def forward(self, patch_features, entity_embeddings):
        projected_patches = F.normalize(self.vision_proj(patch_features), dim=-1)
        projected_entities = F.normalize(self.entity_proj(entity_embeddings), dim=-1)
        scores = projected_patches @ projected_entities.transpose(-1, -2)
        weights = torch.softmax(scores / self.tau, dim=-1)
        aligned = weights @ projected_entities
        return aligned, weights

