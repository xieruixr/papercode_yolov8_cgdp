"""AIFI wrapper module for use after multi-output backbone modules"""
import torch
import torch.nn as nn
from ultralytics.nn.modules.transformer import AIFI as AIFIBase


class AIFIWrapper(nn.Module):
    """
    AIFI wrapper that handles input from multi-output backbones like starnet.
    Takes only the last feature from backbone outputs and applies AIFI transformation.
    Maintains channel count (AIFI is a transformer that doesn't change channels).
    """
    
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize AIFI wrapper.
        Args:
            c1: input/output channels (AIFI preserves channels)
            cm: internal FFN channels
            num_heads: number of attention heads
            dropout: dropout rate
            act: activation function
            normalize_before: whether to apply normalization before transformation
        """
        super().__init__()
        self.aifi = AIFIBase(c1, cm, num_heads, dropout, act, normalize_before)
        self.c1 = c1
        
    def forward(self, x):
        """Forward pass.
        Args:
            x: Input tensor of shape [B, C, H, W] or list of tensors from multi-output backbone
        Returns:
            y: Output tensor of shape [B, C, H, W] with same channels as input
        """
        # If x is a list (from multi-output backbone), take the last element
        if isinstance(x, (list, tuple)):
            x = x[-1]
        
        # Apply AIFI transformer
        return self.aifi(x)
