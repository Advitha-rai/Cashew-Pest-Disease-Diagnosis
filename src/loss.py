"""
Cashew Pest and Disease Diagnosis System
Custom Loss Functions for Class Imbalance (Weighted CrossEntropy & Focal Loss)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha.to(inputs.device) if self.alpha is not None else None)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def get_loss_function(
    class_weights: torch.Tensor = None,
    use_focal_loss: bool = False,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    """
    Automatically selects the optimal loss function based on class balance.
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if use_focal_loss:
        print("[LOSS CONFIG] Selected: Focal Loss (gamma=2.0)")
        return FocalLoss(alpha=class_weights)
    elif class_weights is not None:
        print("[LOSS CONFIG] Selected: Class-Weighted CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        print("[LOSS CONFIG] Selected: Standard CrossEntropyLoss")
        return nn.CrossEntropyLoss()
