import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyWithError(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.last_error = None  # You can access this after forward

    def forward(self, input, target):
        """
        input: [B, C] raw logits
        target: [B] integer class labels
        """
        # Compute softmax probabilities
        probs = F.softmax(input, dim=1)
        
        # One-hot encode targets
        one_hot = F.one_hot(target, num_classes=input.size(1)).float()
        
        # Compute and store the error
        self.last_error = probs - one_hot

        # Return standard cross entropy loss
        return F.cross_entropy(input, target, reduction=self.reduction)
