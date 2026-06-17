import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


class WeightedFusion2(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.eps = eps

    def forward(self, x1, x2):
        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.eps)
        return w[0] * x1 + w[1] * x2


class WeightedFusion3(nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32))
        self.eps = eps

    def forward(self, x1, x2, x3):
        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.eps)
        return w[0] * x1 + w[1] * x2 + w[2] * x3


class DynamicFusion2(nn.Module):
    """Dynamic weighted fusion for 2 inputs."""
    def __init__(self, c, r=4):
        super().__init__()
        hidden = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c * 2, hidden, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 2, 1, 1, 0, bias=True)

    def forward(self, x1, x2):
        g1 = self.pool(x1)
        g2 = self.pool(x2)
        g = torch.cat([g1, g2], dim=1)
        w = self.fc2(self.act(self.fc1(g)))  # [B, 2, 1, 1]
        w = torch.softmax(w, dim=1)

        return x1 * w[:, 0:1] + x2 * w[:, 1:2]


class DynamicFusion3(nn.Module):
    """Dynamic weighted fusion for 3 inputs."""
    def __init__(self, c, r=4):
        super().__init__()
        hidden = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c * 3, hidden, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 3, 1, 1, 0, bias=True)

    def forward(self, x1, x2, x3):
        g1 = self.pool(x1)
        g2 = self.pool(x2)
        g3 = self.pool(x3)
        g = torch.cat([g1, g2, g3], dim=1)
        w = self.fc2(self.act(self.fc1(g)))  # [B, 3, 1, 1]
        w = torch.softmax(w, dim=1)

        return x1 * w[:, 0:1] + x2 * w[:, 1:2] + x3 * w[:, 2:3]


class LiteRefine(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = Conv(c, c, 3, 1, g=c)
        self.pw = Conv(c, c, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


class LABFNeck(nn.Module):
    """
    Lightweight Adaptive Bidirectional Fusion Neck
    Input: [P3, P4, P5]
    Output: [N3, N4, N5]

    Args:
        c3, c4, c5: input channels of P3/P4/P5.
        out_c: int or (c_out3, c_out4, c_out5).
        inner_c: internal fusion channels. If None, uses out_c when out_c is int,
            or max(out_c) when out_c is a sequence.
    """
    def __init__(self, c3, c4, c5, out_c=256, inner_c=None):
        super().__init__()

        if isinstance(out_c, (list, tuple)):
            if len(out_c) != 3:
                raise ValueError(f'out_c as sequence must have length 3, got {len(out_c)}')
            out3, out4, out5 = [int(v) for v in out_c]
            fusion_c = int(inner_c) if inner_c is not None else max(out3, out4, out5)
        else:
            fusion_c = int(inner_c) if inner_c is not None else int(out_c)
            out3 = out4 = out5 = int(out_c)

        # channel align
        self.cv3 = Conv(c3, fusion_c, 1, 1)
        self.cv4 = Conv(c4, fusion_c, 1, 1)
        self.cv5 = Conv(c5, fusion_c, 1, 1)

        # resample
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down3 = Conv(fusion_c, fusion_c, 3, 2)
        self.down4 = Conv(fusion_c, fusion_c, 3, 2)

        # top-down fusion
        self.fuse_t4 = WeightedFusion2()
        self.fuse_t3 = WeightedFusion2()

        # bottom-up fusion
        self.fuse_n4 = WeightedFusion3()
        self.fuse_n5 = WeightedFusion3()

        # refinement
        self.refine_t4 = LiteRefine(fusion_c)
        self.refine_t3 = LiteRefine(fusion_c)
        self.refine_n4 = LiteRefine(fusion_c)
        self.refine_n5 = LiteRefine(fusion_c)

        # output projection (identity if channels are unchanged)
        self.out3 = nn.Identity() if out3 == fusion_c else Conv(fusion_c, out3, 1, 1)
        self.out4 = nn.Identity() if out4 == fusion_c else Conv(fusion_c, out4, 1, 1)
        self.out5 = nn.Identity() if out5 == fusion_c else Conv(fusion_c, out5, 1, 1)

    def forward(self, x):
        p3, p4, p5 = x

        q3 = self.cv3(p3)
        q4 = self.cv4(p4)
        q5 = self.cv5(p5)

        # top-down
        t5 = q5
        t4 = self.refine_t4(self.fuse_t4(q4, self.up(t5)))
        t3 = self.refine_t3(self.fuse_t3(q3, self.up(t4)))

        # bottom-up
        n3 = t3
        n4 = self.refine_n4(self.fuse_n4(q4, t4, self.down3(n3)))
        n5 = self.refine_n5(self.fuse_n5(q5, t5, self.down4(n4)))

        return [self.out3(n3), self.out4(n4), self.out5(n5)]
    

class DLABFNeck(nn.Module):
    """
    Lightweight Adaptive Bidirectional Fusion Neck
    Input: [P3, P4, P5]
    Output: [N3, N4, N5]

    Args:
        c3, c4, c5: input channels of P3/P4/P5.
        out_c: int or (c_out3, c_out4, c_out5).
        inner_c: internal fusion channels. If None, uses out_c when out_c is int,
            or max(out_c) when out_c is a sequence.
    """
    def __init__(self, c3, c4, c5, out_c=256, inner_c=None):
        super().__init__()

        if isinstance(out_c, (list, tuple)):
            if len(out_c) != 3:
                raise ValueError(f'out_c as sequence must have length 3, got {len(out_c)}')
            out3, out4, out5 = [int(v) for v in out_c]
            fusion_c = int(inner_c) if inner_c is not None else max(out3, out4, out5)
        else:
            fusion_c = int(inner_c) if inner_c is not None else int(out_c)
            out3 = out4 = out5 = int(out_c)

        # channel align
        self.cv3 = Conv(c3, fusion_c, 1, 1)
        self.cv4 = Conv(c4, fusion_c, 1, 1)
        self.cv5 = Conv(c5, fusion_c, 1, 1)

        # resample
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down3 = Conv(fusion_c, fusion_c, 3, 2)
        self.down4 = Conv(fusion_c, fusion_c, 3, 2)

        # top-down fusion
        self.fuse_t4 = DynamicFusion2(fusion_c, r=4)
        self.fuse_t3 = DynamicFusion2(fusion_c, r=4)

        # bottom-up fusion
        self.fuse_n4 = DynamicFusion3(fusion_c, r=4)
        self.fuse_n5 = DynamicFusion3(fusion_c, r=4)

        # refinement
        self.refine_t4 = LiteRefine(fusion_c)
        self.refine_t3 = LiteRefine(fusion_c)
        self.refine_n4 = LiteRefine(fusion_c)
        self.refine_n5 = LiteRefine(fusion_c)

        # output projection (identity if channels are unchanged)
        self.out3 = nn.Identity() if out3 == fusion_c else Conv(fusion_c, out3, 1, 1)
        self.out4 = nn.Identity() if out4 == fusion_c else Conv(fusion_c, out4, 1, 1)
        self.out5 = nn.Identity() if out5 == fusion_c else Conv(fusion_c, out5, 1, 1)

    def forward(self, x):
        p3, p4, p5 = x

        q3 = self.cv3(p3)
        q4 = self.cv4(p4)
        q5 = self.cv5(p5)

        # top-down
        t5 = q5
        t4 = self.refine_t4(self.fuse_t4(q4, self.up(t5)))
        t3 = self.refine_t3(self.fuse_t3(q3, self.up(t4)))

        # bottom-up
        n3 = t3
        n4 = self.refine_n4(self.fuse_n4(q4, t4, self.down3(n3)))
        n5 = self.refine_n5(self.fuse_n5(q5, t5, self.down4(n4)))

        return [self.out3(n3), self.out4(n4), self.out5(n5)]
    
class DynamicTripleFusion(nn.Module):
    """
    Dynamic triple-input fusion module for replacing concat in neck.
    Inputs must have same spatial size.
    """
    def __init__(self, c1, c2, c3, cout, r=4):
        super().__init__()
        hidden = max(cout // r, 8)

        # channel align
        self.cv1 = Conv(c1, cout, 1, 1)
        self.cv2 = Conv(c2, cout, 1, 1)
        self.cv3 = Conv(c3, cout, 1, 1)

        # dynamic weight generator
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(cout * 3, hidden, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 3, 1, 1, 0, bias=True)

        # residual preserve branch
        self.res_conv = Conv(cout * 3, cout, 1, 1)

        # refine
        self.refine = Conv(cout, cout, 3, 1, g=cout)

    def forward(self, x):
        x1, x2, x3 = x

        u1 = self.cv1(x1)
        u2 = self.cv2(x2)
        u3 = self.cv3(x3)

        # dynamic weights
        g1 = self.pool(u1)
        g2 = self.pool(u2)
        g3 = self.pool(u3)
        g = torch.cat([g1, g2, g3], dim=1)

        w = self.fc2(self.act(self.fc1(g)))   # [B, 3, 1, 1]
        w = torch.softmax(w, dim=1)

        fdyn = u1 * w[:, 0:1] + u2 * w[:, 1:2] + u3 * w[:, 2:3]

        # residual preserve
        fres = self.res_conv(torch.cat([u1, u2, u3], dim=1))

        out = self.refine(fdyn + fres)
        return out


class OffsetAlign(nn.Module):
    """Offset-guided alignment from a lower-resolution semantic feature to a higher-resolution target."""

    def __init__(self, c_high, c_low, c_out, hidden=64, offset_scale=0.5):
        super().__init__()
        self.offset_scale = offset_scale
        hidden = max(int(hidden), 16)

        self.high_proj = Conv(c_high, c_out, 1, 1)
        self.low_proj = Conv(c_low, c_out, 1, 1)
        self.offset = nn.Sequential(
            Conv(c_out * 2, hidden, 3, 1),
            nn.Conv2d(hidden, 2, 1, 1, 0, bias=True),
        )
        self.out_proj = Conv(c_out, c_out, 1, 1)

    @staticmethod
    def _base_grid(b, h, w, device, dtype):
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device, dtype=dtype),
            torch.linspace(-1, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xs, ys), dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)

    def forward(self, x_high, x_low):
        xh = self.high_proj(x_high)
        xl = self.low_proj(x_low)
        xh = F.interpolate(xh, size=xl.shape[-2:], mode="bilinear", align_corners=False)

        offset = self.offset(torch.cat([xh, xl], dim=1))
        b, _, h, w = xh.shape
        base_grid = self._base_grid(b, h, w, xh.device, xh.dtype)

        offset_x = offset[:, 0] / max(w - 1, 1) * 2.0
        offset_y = offset[:, 1] / max(h - 1, 1) * 2.0
        sampling_grid = base_grid + self.offset_scale * torch.stack([offset_x, offset_y], dim=-1)

        xh = F.grid_sample(
            xh,
            sampling_grid.clamp(-1.2, 1.2),
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return self.out_proj(xh), xl


class LargeKernelSelect(nn.Module):
    """Lightweight large-kernel selective context refinement."""

    def __init__(self, c):
        super().__init__()
        self.dw5 = nn.Conv2d(c, c, kernel_size=5, padding=2, groups=c, bias=False)
        self.dw7d3 = nn.Conv2d(c, c, kernel_size=7, padding=9, dilation=3, groups=c, bias=False)
        self.pw1 = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.pw2 = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.selector = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(c, c, 1, 1)

    def forward(self, x):
        b1 = self.pw1(self.dw5(x))
        b2 = self.pw2(self.dw7d3(x))
        u = torch.cat([b1, b2], dim=1)
        s = torch.cat([u.mean(1, keepdim=True), u.max(1, keepdim=True)[0]], dim=1)
        a = self.selector(s)
        return self.out(b1 * a[:, 0:1] + b2 * a[:, 1:2] + x)


class FreqAwareGate(nn.Module):
    """Frequency-compensated residual refinement for boundary/detail preservation."""

    def __init__(self, c, hidden_ratio=0.5):
        super().__init__()
        hidden = max(int(c * hidden_ratio), 16)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.hf_branch = nn.Sequential(
            Conv(c, hidden, 1, 1),
            Conv(hidden, hidden, 3, 1, g=hidden),
            Conv(hidden, c, 1, 1, act=False),
        )
        self.ctx_branch = nn.Sequential(
            Conv(c, hidden, 1, 1),
            Conv(hidden, hidden, 5, 1, g=hidden),
            Conv(hidden, c, 1, 1, act=False),
        )
        gate_hidden = max(c // 16, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, gate_hidden, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 2, 1, 1, 0, bias=True),
            nn.Softmax(dim=1),
        )
        self.out = Conv(c, c, 1, 1)

    def forward(self, x):
        x_hf = x - self.pool(x)
        f_hf = self.hf_branch(x_hf)
        f_ctx = self.ctx_branch(x)
        w = self.gate(x)
        return x + self.out(w[:, 0:1] * f_hf + w[:, 1:2] * f_ctx)


class ScaleRouteFusion3(nn.Module):
    """Instance-adaptive three-branch routing fusion."""

    def __init__(self, c, r=4):
        super().__init__()
        hidden = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c * 3, hidden, 1, 1, 0, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, 3, 1, 1, 0, bias=True)
        self.mix = Conv(c * 3, c, 1, 1)

    def forward(self, x1, x2, x3):
        g = torch.cat([self.pool(x1), self.pool(x2), self.pool(x3)], dim=1)
        w = torch.softmax(self.fc2(self.act(self.fc1(g))), dim=1)
        routed = x1 * w[:, 0:1] + x2 * w[:, 1:2] + x3 * w[:, 2:3]
        preserved = self.mix(torch.cat([x1, x2, x3], dim=1))
        return routed + preserved


class AARFNode(nn.Module):
    """
    Alignment-Aware Routing and Frequency node.
    Inspired by dynamic level interaction, routed sparse selection, and frequency-aware fusion.
    """

    def __init__(self, c_high, c_low, c_out, hidden=64, route_r=4):
        super().__init__()
        self.align = OffsetAlign(c_high, c_low, c_out, hidden=hidden)
        self.low_proj = Conv(c_low, c_out, 1, 1)
        self.route = ScaleRouteFusion3(c_out, r=route_r)
        self.freq = FreqAwareGate(c_out)
        self.lks = LargeKernelSelect(c_out)

    def forward(self, x_high, x_low, x_skip):
        xa, xl = self.align(x_high, x_low)
        xs = self.low_proj(x_skip)
        x = self.route(xa, xl, xs)
        x = self.freq(x)
        return self.lks(x)


class DownFusionNode(nn.Module):
    """Bottom-up tri-source fusion with detail compensation and large-kernel context."""

    def __init__(self, c_curr, c_skip, c_sem, c_out, route_r=4):
        super().__init__()
        self.curr = Conv(c_curr, c_out, 3, 2)
        self.skip = Conv(c_skip, c_out, 1, 1)
        self.sem = Conv(c_sem, c_out, 1, 1)
        self.route = ScaleRouteFusion3(c_out, r=route_r)
        self.freq = FreqAwareGate(c_out)
        self.lks = LargeKernelSelect(c_out)

    def forward(self, x_curr, x_skip, x_sem):
        xc = self.curr(x_curr)
        xs = self.skip(x_skip)
        xm = self.sem(x_sem)
        x = self.route(xc, xs, xm)
        x = self.freq(x)
        return self.lks(x)


class AARFNeck(nn.Module):
    """
    Alignment-Aware Routing and Frequency Neck.

    Design summary:
    1) offset-guided semantic alignment for cross-scale fusion
    2) routed tri-branch fusion to avoid fixed concat-heavy aggregation
    3) frequency-compensated refinement for boundary preservation
    4) large-kernel selective context to enlarge effective receptive field
    """

    def __init__(self, c3, c4, c5, out_c=256, inner_c=None, hidden=64):
        super().__init__()

        if isinstance(out_c, (list, tuple)):
            if len(out_c) != 3:
                raise ValueError(f"out_c as sequence must have length 3, got {len(out_c)}")
            out3, out4, out5 = [int(v) for v in out_c]
            fusion_c = int(inner_c) if inner_c is not None else max(out3, out4, out5)
        else:
            fusion_c = int(inner_c) if inner_c is not None else int(out_c)
            out3 = out4 = out5 = int(out_c)

        self.p3 = Conv(c3, fusion_c, 1, 1)
        self.p4 = Conv(c4, fusion_c, 1, 1)
        self.p5 = Conv(c5, fusion_c, 1, 1)

        self.td4 = AARFNode(fusion_c, fusion_c, fusion_c, hidden=hidden)
        self.td3 = AARFNode(fusion_c, fusion_c, fusion_c, hidden=hidden)

        self.out3_refine = nn.Sequential(FreqAwareGate(fusion_c), LargeKernelSelect(fusion_c))
        self.bu4 = DownFusionNode(fusion_c, fusion_c, fusion_c, fusion_c)
        self.bu5 = DownFusionNode(fusion_c, fusion_c, fusion_c, fusion_c)

        self.out3 = nn.Identity() if out3 == fusion_c else Conv(fusion_c, out3, 1, 1)
        self.out4 = nn.Identity() if out4 == fusion_c else Conv(fusion_c, out4, 1, 1)
        self.out5 = nn.Identity() if out5 == fusion_c else Conv(fusion_c, out5, 1, 1)

    def forward(self, x):
        p3, p4, p5 = x
        q3 = self.p3(p3)
        q4 = self.p4(p4)
        q5 = self.p5(p5)

        t5 = q5
        t4 = self.td4(t5, q4, q4)
        t3 = self.td3(t4, q3, q3)

        n3 = self.out3_refine(t3)
        n4 = self.bu4(n3, q4, t4)
        n5 = self.bu5(n4, q5, t5)

        return [self.out3(n3), self.out4(n4), self.out5(n5)]
