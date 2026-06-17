import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class DWConv(nn.Module):
    """Depthwise separable convolution in YOLO-style."""
    def __init__(self, c1, c2, k=3, s=1, act=True):
        super().__init__()
        self.dw = Conv(c1, c1, k, s, g=c1, act=act)
        self.pw = Conv(c1, c2, 1, 1, g=1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class GDDWeightedGather(nn.Module):
    """
    Static learnable weighted gather:
        g = w3 * g3 + w4 * g4 + w5 * g5
    """
    def __init__(self, n=3, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n, dtype=torch.float32))
        self.eps = eps

    def forward(self, feats):
        # feats: list of pooled descriptors, each [B, C, 1, 1]
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = 0.0
        for i, f in enumerate(feats):
            out = out + w[i] * f
        return out


class GDDChannelGate(nn.Module):
    """
    Channel gate for distribute stage.
    """
    def __init__(self, c, r=4):
        super().__init__()
        hidden = max(c // r, 8)
        self.cv1 = nn.Conv2d(c, hidden, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.cv2 = nn.Conv2d(hidden, c, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, 1, 1]
        return self.sigmoid(self.cv2(self.act(self.cv1(x))))


class GDDSpatialGate(nn.Module):
    """
    Optional lightweight spatial gate.
    """
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, 1, 3, 1, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        return self.sigmoid(self.conv(x))


class GDDRefine(nn.Module):
    """
    Lightweight local refinement after distribute.
    """
    def __init__(self, c):
        super().__init__()
        self.block = DWConv(c, c, 3, 1)

    def forward(self, x):
        return self.block(x)


class GDDNeck(nn.Module):
    """
    Gather-Distribute Dynamic Neck block for 3-scale features.

    This module is designed to be inserted after an existing neck, e.g.:
        backbone -> PAN/FPN/BiFPN -> GDDNeck -> Detect

    Args:
        c3 (int): input channels of P3
        c4 (int): input channels of P4
        c5 (int): input channels of P5
        out_c (int): unified output channels
        r (int): reduction ratio of channel gate
        spatial (bool): whether to use spatial gate
        shortcut (bool): whether to use residual gating form
    Inputs:
        x = [p3, p4, p5]
    Outputs:
        [o3, o4, o5]
    """
    def __init__(self, c3, c4, c5, out_c=256, r=4, spatial=False, shortcut=True, restore_channels=False):
        super().__init__()
        if isinstance(out_c, (list, tuple)):
            if len(out_c) != 3:
                raise ValueError(f"GDDNeck out_c must be int or length-3 list/tuple, but got {out_c}")
            out_channels = [int(out_c[0]), int(out_c[1]), int(out_c[2])]
            mid_c = max(out_channels)
        else:
            mid_c = int(out_c)
            out_channels = [mid_c, mid_c, mid_c]

        self.mid_c = mid_c
        self.out_channels = out_channels
        self.spatial = spatial
        self.shortcut = shortcut
        self.restore_channels = restore_channels

        # Channel alignment
        self.reduce_p3 = Conv(c3, mid_c, k=1, s=1)
        self.reduce_p4 = Conv(c4, mid_c, k=1, s=1)
        self.reduce_p5 = Conv(c5, mid_c, k=1, s=1)

        # Gather
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gather = GDDWeightedGather(n=3)

        # Distribute gates
        self.gate_p3 = GDDChannelGate(mid_c, r)
        self.gate_p4 = GDDChannelGate(mid_c, r)
        self.gate_p5 = GDDChannelGate(mid_c, r)

        if spatial:
            self.sgate_p3 = GDDSpatialGate(mid_c)
            self.sgate_p4 = GDDSpatialGate(mid_c)
            self.sgate_p5 = GDDSpatialGate(mid_c)

        # Refinement
        self.refine_p3 = GDDRefine(mid_c)
        self.refine_p4 = GDDRefine(mid_c)
        self.refine_p5 = GDDRefine(mid_c)

        # Optional restore to original Detect input channels.
        if restore_channels:
            self.restore_p3 = Conv(mid_c, c3, k=1, s=1, act=False) if mid_c != c3 else nn.Identity()
            self.restore_p4 = Conv(mid_c, c4, k=1, s=1, act=False) if mid_c != c4 else nn.Identity()
            self.restore_p5 = Conv(mid_c, c5, k=1, s=1, act=False) if mid_c != c5 else nn.Identity()
        else:
            self.restore_p3 = Conv(mid_c, out_channels[0], k=1, s=1, act=False) if mid_c != out_channels[0] else nn.Identity()
            self.restore_p4 = Conv(mid_c, out_channels[1], k=1, s=1, act=False) if mid_c != out_channels[1] else nn.Identity()
            self.restore_p5 = Conv(mid_c, out_channels[2], k=1, s=1, act=False) if mid_c != out_channels[2] else nn.Identity()

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, "GDDNeck expects [P3, P4, P5]"
        p3, p4, p5 = x

        # 1) align channels
        q3 = self.reduce_p3(p3)
        q4 = self.reduce_p4(p4)
        q5 = self.reduce_p5(p5)

        # 2) gather global multi-scale descriptor
        g3 = self.pool(q3)
        g4 = self.pool(q4)
        g5 = self.pool(q5)
        g = self.gather([g3, g4, g5])  # [B, C, 1, 1]

        # 3) distribute channel gating
        w3 = self.gate_p3(g)
        w4 = self.gate_p4(g)
        w5 = self.gate_p5(g)

        if self.shortcut:
            q3 = q3 * w3 + q3
            q4 = q4 * w4 + q4
            q5 = q5 * w5 + q5
        else:
            q3 = q3 * w3
            q4 = q4 * w4
            q5 = q5 * w5

        # 4) optional spatial gating
        if self.spatial:
            s3 = self.sgate_p3(q3)
            s4 = self.sgate_p4(q4)
            s5 = self.sgate_p5(q5)

            if self.shortcut:
                q3 = q3 * s3 + q3
                q4 = q4 * s4 + q4
                q5 = q5 * s5 + q5
            else:
                q3 = q3 * s3
                q4 = q4 * s4
                q5 = q5 * s5

        # 5) local refinement
        o3 = self.restore_p3(self.refine_p3(q3))
        o4 = self.restore_p4(self.refine_p4(q4))
        o5 = self.restore_p5(self.refine_p5(q5))

        return [o3, o4, o5]
