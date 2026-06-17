import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["Conv", "DWConv", "DyAlignUp", "R2SF", "FCRB", "RDAFNode"]


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        if isinstance(k, int):
            k = d * (k - 1) + 1
        else:
            k = [d * (x - 1) + 1 for x in k]
    if p is None:
        if isinstance(k, int):
            p = k // 2
        else:
            p = [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BN and SiLU."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=3, s=1, d=1, act=True):
        super().__init__()
        g = math.gcd(c1, c2)
        self.conv = Conv(c1, c2, k, s, g=g, d=d, act=act)

    def forward(self, x):
        return self.conv(x)


class DyAlignUp(nn.Module):
    """
    Dynamic alignment upsampling.

    Input:
        x = [x_high, x_low]
            x_high: [B, C_high, H/2, W/2] or lower-res feature
            x_low : [B, C_low,  H,   W  ] target-resolution feature

    Output:
        aligned high-level feature with shape [B, C_out, H, W]
    """

    def __init__(self, c_high, c_low, c_out, hidden=64, offset_scale=0.5):
        super().__init__()
        self.offset_scale = offset_scale

        self.high_proj = Conv(c_high, c_out, k=1, s=1)
        self.low_proj = Conv(c_low, c_out, k=1, s=1)

        self.offset_net = nn.Sequential(
            Conv(c_out * 2, hidden, k=3, s=1),
            nn.Conv2d(hidden, 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.out_proj = Conv(c_out, c_out, k=1, s=1)

    @staticmethod
    def _make_base_grid(b, h, w, device, dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        grid = torch.stack((xs, ys), dim=-1)  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(b, 1, 1, 1)  # [B, H, W, 2]
        return grid

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DyAlignUp expects input as [x_high, x_low].")

        x_high, x_low = x

        # project features
        xh = self.high_proj(x_high)
        xl = self.low_proj(x_low)

        # resize high-level feature to low-level spatial size
        xh = F.interpolate(xh, size=xl.shape[-2:], mode="bilinear", align_corners=False)

        # predict offsets conditioned on both scales
        offset = self.offset_net(torch.cat([xh, xl], dim=1))  # [B, 2, H, W]

        b, c, h, w = xh.shape
        base_grid = self._make_base_grid(b, h, w, xh.device, xh.dtype)

        # normalize offsets into [-1, 1] grid space
        offset_x = offset[:, 0] / max(w - 1, 1) * 2.0
        offset_y = offset[:, 1] / max(h - 1, 1) * 2.0
        offset_grid = torch.stack([offset_x, offset_y], dim=-1)  # [B, H, W, 2]

        sampling_grid = base_grid + self.offset_scale * offset_grid
        sampling_grid = sampling_grid.clamp(-1.2, 1.2)

        xh_aligned = F.grid_sample(
            xh,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        return self.out_proj(xh_aligned)


class R2SF(nn.Module):
    """
    Region-routed scale fusion.

    Input:
        x = [x1, x2]
            x1: aligned high-level feature [B, C1, H, W]
            x2: low-level feature         [B, C2, H, W]

    Output:
        fused feature [B, C_out, H, W]

    Notes:
        This is an engineering-friendly approximation of region routing:
        - compute region descriptors by adaptive pooling
        - obtain cross-region similarity
        - keep top-k scores
        - generate a pixel-level routing score map
        - suppress less relevant regions before fusion
    """

    def __init__(self, c1, c2, c_out, region_size=4, topk=2, hidden=None):
        super().__init__()
        self.region_size = max(int(region_size), 1)
        self.topk = max(int(topk), 1)

        self.proj1 = Conv(c1, c_out, k=1, s=1)
        self.proj2 = Conv(c2, c_out, k=1, s=1)

        h = hidden if hidden is not None else max(c_out // 2, 16)

        self.score_refine = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            Conv(c_out * 2, c_out, k=1, s=1),
            DWConv(c_out, c_out, k=3, s=1),
            Conv(c_out, c_out, k=1, s=1),
        )

        # optional lightweight residual preserve
        self.preserve = Conv(c_out, c_out, k=1, s=1, act=False)

    def _region_descriptor(self, x):
        """
        Adaptive region pooling.
        Output shape: [B, C, gh, gw]
        """
        b, c, h, w = x.shape
        gh = max(h // self.region_size, 1)
        gw = max(w // self.region_size, 1)
        return F.adaptive_avg_pool2d(x, output_size=(gh, gw))

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("R2SF expects input as [x1, x2].")

        x1, x2 = x
        x1 = self.proj1(x1)
        x2 = self.proj2(x2)

        q = self._region_descriptor(x1)  # [B, C, gh, gw]
        k = self._region_descriptor(x2)  # [B, C, gh, gw]

        b, c, gh, gw = q.shape
        n = gh * gw

        qf = q.flatten(2).transpose(1, 2)  # [B, N, C]
        kf = k.flatten(2).transpose(1, 2)  # [B, N, C]

        sim = torch.matmul(qf, kf.transpose(-1, -2)) / math.sqrt(max(c, 1))  # [B, N, N]

        k_top = min(self.topk, sim.shape[-1])
        topk_vals, _ = sim.topk(k=k_top, dim=-1)  # [B, N, topk]

        # per-query region routed score
        routed_score = topk_vals.mean(dim=-1)  # [B, N]
        routed_score = routed_score.view(b, 1, gh, gw)

        # normalize to [0, 1]
        routed_score = routed_score - routed_score.amin(dim=(2, 3), keepdim=True)
        routed_score = routed_score / (routed_score.amax(dim=(2, 3), keepdim=True) + 1e-6)
        routed_score = self.score_refine(routed_score)

        # upsample routing score to pixel level
        routed_score = F.interpolate(routed_score, size=x1.shape[-2:], mode="nearest")

        # suppress noisy regions from low-level feature before fusion
        x2r = x2 * routed_score

        out = self.fuse(torch.cat([x1, x2r], dim=1))
        out = out + self.preserve(x1)
        return out


class FCRB(nn.Module):
    """
    Frequency-compensated residual block.

    Input:
        x: [B, C, H, W]

    Output:
        refined x with same shape
    """

    def __init__(self, c, hidden_ratio=0.5, use_gate=True, pool_k=3, large_k=5):
        super().__init__()
        hidden = max(int(c * hidden_ratio), 16)
        self.use_gate = use_gate

        self.pool = nn.AvgPool2d(kernel_size=pool_k, stride=1, padding=pool_k // 2)

        # local high-frequency branch
        self.local_branch = nn.Sequential(
            Conv(c, hidden, k=1, s=1),
            DWConv(hidden, hidden, k=3, s=1),
            Conv(hidden, c, k=1, s=1, act=False),
        )

        # wide context branch
        self.context_branch = nn.Sequential(
            Conv(c, hidden, k=1, s=1),
            DWConv(hidden, hidden, k=large_k, s=1),
            Conv(hidden, c, k=1, s=1, act=False),
        )

        if self.use_gate:
            gate_hidden = max(c // 16, 8)
            self.gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c, gate_hidden, kernel_size=1, stride=1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_hidden, 2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.Softmax(dim=1),
            )
        else:
            self.gate = None

        self.out = Conv(c, c, k=1, s=1)

    def forward(self, x):
        x_hf = x - self.pool(x)

        f_local = self.local_branch(x_hf)
        f_ctx = self.context_branch(x)

        if self.use_gate:
            w = self.gate(x)  # [B, 2, 1, 1]
            f = w[:, 0:1] * f_local + w[:, 1:2] * f_ctx
        else:
            f = 0.5 * f_local + 0.5 * f_ctx

        return x + self.out(f)


class RDAFNode(nn.Module):
    """
    Region-routed Dynamic Alignment and Frequency-compensated fusion node.

    Input:
        x = [x_high, x_low]
            x_high: low-resolution high-semantic feature
            x_low : high-resolution low/mid-level feature

    Output:
        y: fused feature at x_low resolution

    Recommended YAML usage:
        - [[high_idx, low_idx], 1, RDAFNode, [c_high, c_low, c_out]]

    Example:
        - [[-1, 6], 1, RDAFNode, [512, 512, 512]]

    Args:
        c_high: channels of x_high
        c_low : channels of x_low
        c_out : output channels
        hidden: hidden channels for DyAlignUp offset net
        region_size: routing region size
        topk: top-k routed regions
        hidden_ratio: hidden ratio in FCRB
        use_fcrb: whether to enable FCRB
    """

    def __init__(
        self,
        c_high,
        c_low,
        c_out,
        hidden=64,
        region_size=4,
        topk=2,
        hidden_ratio=0.5,
        use_fcrb=True,
    ):
        super().__init__()
        self.align = DyAlignUp(c_high, c_low, c_out, hidden=hidden)
        self.r2sf = R2SF(c_out, c_low, c_out, region_size=region_size, topk=topk)
        self.use_fcrb = use_fcrb
        self.fcrb = FCRB(c_out, hidden_ratio=hidden_ratio, use_gate=True) if use_fcrb else nn.Identity()

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("RDAFNode expects input as [x_high, x_low].")

        x_high, x_low = x
        xh = self.align([x_high, x_low])
        xf = self.r2sf([xh, x_low])
        y = self.fcrb(xf)
        return y