# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from ultralytics.utils.torch_utils import fuse_conv_and_bn
import math
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad,ADRConv
from .transformer import TransformerBlock
from einops import rearrange, reduce
from .esc import ESCBlock, ConvAttn
from .attention import *
from .dynamic_snake_conv import DySnakeConv
from timm.layers import to_2tuple
from functools import partial
from .metaformer import *
__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "SBA",
    "SPDConv",
    "SBA_DySample",
    "CSPOmniKernel",
    "DySample",
    "LAWDS",
    "C2f_ESC",
    "Bottleneck_ConvAttn",
    "C2f_ConvAttn",
    "LSKNet",
    "EnhancedSBA",
    "C2f_DCNv4",
    "SPPF_LSKA",
    "C2f_DySnakeConv",
    "DyHeadBlock",
    "DyHeadBlockWithDCNV4",
    "C2f_EIEM",
    "C2f_SMAFB_CGLU",
    "C2f_SMPCGLU",
    "CSP_MutilScaleEdgeInformationSelect",
    "C2f_DCMB",
    "C2f_DCMB_KAN",
    "ConvEdgeFusion",
    "MutilScaleEdgeInfoGenetator",
    "C2f_IdentityFormer",
    "C2f_RandomMixing",
    "C2f_PoolingFormer",
    "C2f_ConvFormer",
    "C2f_CaFormer",
    "C2f_IdentityFormerCGLU",
    "C2f_RandomMixingCGLU",
    "C2f_PoolingFormerCGLU",
    "C2f_ConvFormerCGLU",
    "C2f_CaFormerCGLU",
    "C2f_AP",
    "RGFM",
    "C2f_Star",
    "FocusFeature",
    "SGRNeck",
    "C2f_SGR",
    "C2f_HKSM",
    "FeaturePyramidSharedConv",
    "Spectra_SPPF",
    "DSR_C2f",
    "SpectraLite_SPPF",
    "RDS_C2f",
    "SpectraEdge_SPPF",
    "RDEA_C2f",
    "C2f_StarAlign",
    "C2f_AINLite",
    "C2f_AINLite_ECA",
    "MetaSPPF_LSKA_CGLU",
    "C2f_MetaStripCGLU",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSA(nn.Module):
    """
    Position-wise Spatial Attention module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        e (float): Expansion factor for the intermediate channels. Default is 0.5.

    Attributes:
        c (int): Number of intermediate channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for spatial attention.
        ffn (nn.Sequential): Feed-forward network module.
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes convolution layers, attention module, and feed-forward network with channel reduction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """
        Forward pass of the PSA module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class SCDown(nn.Module):
    """Spatial Channel Downsample (SCDown) module for reducing spatial and channel dimensions."""

    def __init__(self, c1, c2, k, s):
        """
        Spatial Channel Downsample (SCDown) module.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for the convolutional layer.
            s (int): Stride for the convolutional layer.
        """
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """
        Forward pass of the SCDown module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the SCDown module.
        """
        return self.cv2(self.cv1(x))


######################################## Re-CalibrationFPN start ########################################

def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class SBA(nn.Module):

    def __init__(self, inc, input_dim=64):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
                
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        
        self.Sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * Upsample(g_H_feature * H_feature, size= L_feature.size()[2:], align_corners=False)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * Upsample(g_L_feature * L_feature, size= H_feature.size()[2:], align_corners=False) 
        
        H_feature = Upsample(H_feature, size = L_feature.size()[2:])
        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out

######################################## Re-CalibrationFPN end ########################################

######################################## Re-CalibrationFPN_DySample start ########################################

def Upsample(x, size, align_corners = False):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=align_corners)

class SBA_DySample(nn.Module):

    def __init__(self, inc, input_dim=64, dysample_groups=2):
        super().__init__()

        self.input_dim = input_dim

        self.d_in1 = Conv(input_dim//2, input_dim//2, 1) 
        self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
                
        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        # self.fc3 = ADRConv(input_dim//2, input_dim//2, 3, 2, 1)
        self.fc3 = nn.Conv2d(input_dim//2, input_dim//2, kernel_size=3, stride=2, padding=1)

        self.Sigmoid = nn.Sigmoid()
        
        self.dysample = DySample(input_dim//2, scale=2, style='lp', groups=dysample_groups)

    def fuse_features(self, src, target):
        # src: 要融合的特征
        # target: 目标特征（决定融合方式）
        if src.shape[3] > target.shape[3]:
            # src宽度更大，使用conv下采样
            out = self.fc3(src)
            # 对齐空间尺寸
            return out
        else:
            # src宽度更小或相等，使用dysample上采样
            out = self.dysample(src)
            return out

    def forward(self, x):

        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        g_L_feature =  self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature = L_feature + L_feature * g_L_feature + (1 - g_L_feature) * self.fuse_features(g_H_feature * H_feature, L_feature)
        H_feature = H_feature + H_feature * g_H_feature + (1 - g_H_feature) * self.fuse_features(g_L_feature * L_feature, H_feature)

        H_feature = self.fuse_features(H_feature, L_feature)

        out = self.conv(torch.cat([H_feature, L_feature], dim=1))
        return out
    
######################################## Re-CalibrationFPN_DySample end ########################################

class RGFM(nn.Module):
    """Recursive gated feature fusion module used by YOLOv8-CGDP."""

    def __init__(self, inc, input_dim=64, dysample_groups=2):
        super().__init__()

        self.input_dim = input_dim
        C = input_dim // 2
        self.groups = dysample_groups

        assert C % self.groups == 0, f"C ({C}) must be divisible by groups ({self.groups})"

        self.d_in1 = Conv(C, C, 1)
        self.d_in2 = Conv(C, C, 1)

        self.conv = Conv(input_dim, input_dim, 3)
        self.fc1 = nn.Conv2d(inc[1], C, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], C, kernel_size=1, bias=False)

        # Downsample high-resolution features with DCNv2 when feature maps need alignment.
        self.K = 3
        self.padding = 1
        self.stride = 2

        k_sq = self.K * self.K
        offset_mask_channels = self.groups * 3 * k_sq  # groups * (2*K*K + K*K)

        self.dcn_offset_mask = nn.Conv2d(
            C,
            offset_mask_channels,
            kernel_size=self.K,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

        self.dcn_conv = ModulatedDeformConv2d(
            C,
            C,
            kernel_size=self.K,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            bias=False,
        )

        nn.init.constant_(self.dcn_offset_mask.weight, 0.0)
        if self.dcn_offset_mask.bias is not None:
            nn.init.constant_(self.dcn_offset_mask.bias, 0.0)

        self.sigmoid = nn.Sigmoid()
        self.dysample = DySample(C, scale=2, style="lp", groups=dysample_groups)

    def fuse_features(self, src, target):
        """Align `src` to the spatial size of `target`."""
        if (src.shape[2] > target.shape[2]) or (src.shape[3] > target.shape[3]):
            k_sq = self.K * self.K
            offset_channels = self.groups * 2 * k_sq
            mask_channels = self.groups * k_sq
            expected = offset_channels + mask_channels

            offset_mask = self.dcn_offset_mask(src)  # [B, expected, H/2, W/2]
            if offset_mask.shape[1] != expected:
                raise RuntimeError(
                    f"offset_mask channels ({offset_mask.shape[1]}) != expected ({expected}). "
                    "Check groups, kernel size, and channel settings."
                )

            offset, mask = torch.split(offset_mask, [offset_channels, mask_channels], dim=1)
            offset = offset.contiguous()
            mask = mask.contiguous().sigmoid()
            return self.dcn_conv(src, offset, mask)
        return self.dysample(src)

    def forward(self, x):
        H_feature, L_feature = x

        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)

        g_L_feature = self.sigmoid(L_feature)
        g_H_feature = self.sigmoid(H_feature)

        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        L_feature_fused = (
            L_feature
            + L_feature * g_L_feature
            + (1 - g_L_feature) * self.fuse_features(g_H_feature * H_feature, L_feature)
        )
        H_feature_fused = (
            H_feature
            + H_feature * g_H_feature
            + (1 - g_H_feature) * self.fuse_features(g_L_feature * L_feature, H_feature)
        )

        H_feature_aligned = self.fuse_features(H_feature_fused, L_feature_fused)

        return self.conv(torch.cat([H_feature_aligned, L_feature_fused], dim=1))


######################################## Enhanced SBA start ########################################

class EnhancedSBA(nn.Module):
    """
    Enhanced SBA模块，集成了DySample和ADRConv
    
    相比原始SBA的改进：
    1. 使用ADRConv替换普通卷积，增强特征提取能力
    2. 使用DySample替换双线性插值，实现学习性上采样
    3. 保持原有的双向注意力机制
    """

    def __init__(self, inc, input_dim=64, use_dysample=True, use_adrconv=False):
        super().__init__()
        
        self.input_dim = input_dim
        self.use_dysample = use_dysample
        self.use_adrconv = use_adrconv

        # 特征处理层 - 可选择使用ADRConv
        if use_adrconv:
            from .conv import ADRConv
            self.d_in1 = ADRConv(input_dim//2, input_dim//2, k=1)
            self.d_in2 = ADRConv(input_dim//2, input_dim//2, k=1)
        else:
            self.d_in1 = Conv(input_dim//2, input_dim//2, 1)
            self.d_in2 = Conv(input_dim//2, input_dim//2, 1)       
        
        # 最终融合层 - 可选择使用ADRConv        
        if use_adrconv:
            from .conv import ADRConv
            self.conv = ADRConv(input_dim, input_dim, k=3)
        else:
            self.conv = Conv(input_dim, input_dim, 3)
            
        # 通道调整层
        self.fc1 = nn.Conv2d(inc[1], input_dim//2, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(inc[0], input_dim//2, kernel_size=1, bias=False)
        
        # DySample上采样模块
        if use_dysample:
            # 为不同的上采样需求创建DySample实例
            self.dysample_h2l = DySample(input_dim//2, scale=2, style='lp', groups=4)
            self.dysample_l2h = DySample(input_dim//2, scale=2, style='lp', groups=4) 
            self.dysample_final = DySample(input_dim//2, scale=2, style='lp', groups=4)
        
        self.Sigmoid = nn.Sigmoid()
        
    def adaptive_upsample(self, x, target_size, dysample_module=None):
        """
        自适应上采样函数
        根据配置选择使用DySample或传统插值
        """
        if self.use_dysample and dysample_module is not None:
            # 计算需要的上采样倍数
            _, _, h, w = x.shape
            target_h, target_w = target_size
            
            # 如果尺寸已经匹配，直接返回
            if h == target_h and w == target_w:
                return x
                
            # 如果是2倍上采样，使用DySample
            if target_h == h * 2 and target_w == w * 2:
                return dysample_module(x)
            else:
                # 其他情况回退到传统插值
                return Upsample(x, size=target_size, align_corners=False)
        else:
            return Upsample(x, size=target_size, align_corners=False)

    def forward(self, x):
        """
        Enhanced SBA前向传播
        """
        H_feature, L_feature = x

        # 通道调整
        L_feature = self.fc1(L_feature)
        H_feature = self.fc2(H_feature)
        
        # 门控信号生成
        g_L_feature = self.Sigmoid(L_feature)
        g_H_feature = self.Sigmoid(H_feature)
        
        # 特征增强处理（使用ADRConv）
        L_feature = self.d_in1(L_feature)
        H_feature = self.d_in2(H_feature)

        # 双向特征融合 - 使用DySample进行学习性上采样
        L_enhanced = (L_feature + 
                     L_feature * g_L_feature + 
                     (1 - g_L_feature) * self.adaptive_upsample(
                         g_H_feature * H_feature, 
                         L_feature.size()[2:], 
                         self.dysample_h2l if self.use_dysample else None
                     ))
        
        H_enhanced = (H_feature + 
                     H_feature * g_H_feature + 
                     (1 - g_H_feature) * self.adaptive_upsample(
                         g_L_feature * L_feature, 
                         H_feature.size()[2:], 
                         self.dysample_l2h if self.use_dysample else None
                     ))
        
        # 最终融合 - 统一尺寸并拼接
        H_upsampled = self.adaptive_upsample(
            H_enhanced, 
            L_enhanced.size()[2:], 
            self.dysample_final if self.use_dysample else None
        )
        
        # 特征拼接和最终卷积（使用ADRConv）
        out = self.conv(torch.cat([H_upsampled, L_enhanced], dim=1))
        return out

######################################## Enhanced SBA end ########################################

######################################## SPD-Conv start ########################################

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x

######################################## SPD-Conv end ########################################


######################################## Omni-Kernel Network for Image Restoration [AAAI-24] start ########################################

class FGM(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1, groups=dim)

        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        # res = x.clone()
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)

        x2_fft = torch.fft.fft2(x2, norm='backward')

        out = x1 * x2_fft

        out = torch.fft.ifft2(out, dim=(-2,-1), norm='backward')
        out = torch.abs(out)

        return out * self.alpha + x * self.beta

class OmniKernel(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()

        ker = 31
        pad = ker // 2
        self.in_conv = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
                    nn.GELU()
                    )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1,ker), padding=(0,pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker,1), padding=(pad,0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)

        self.act = nn.ReLU()

        ### sca ###
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### fca ###
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)

        ### fca ###
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2,-1), norm='backward')
        x_fca = torch.abs(x_fca)

        ### fca ###
        ### sca ###
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        ### sca ###
        x_sca = self.fgm(x_sca)

        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

class CSPOmniKernel(nn.Module):
    def __init__(self, dim, e=0.25):
        super().__init__()
        self.e = e
        self.cv1 = Conv(dim, dim, 1)
        self.cv2 = Conv(dim, dim, 1)
        self.m = OmniKernel(int(dim * self.e))

    def forward(self, x):
        ok_branch, identity = torch.split(self.cv1(x), [int(self.cv1.conv.out_channels * self.e), int(self.cv1.conv.out_channels * (1 - self.e))], dim=1)
        return self.cv2(torch.cat((self.m(ok_branch), identity), 1))

######################################## Omni-Kernel Network for Image Restoration [AAAI-24] end ########################################


######################################## DySample start ########################################

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        self.normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            self.constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def normal_init(self, module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def constant_init(self, module, val, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.constant_(module.weight, val)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

######################################## DySample end ########################################

######################################## LAWDS begin ########################################

class LAWDS(nn.Module):
    # Light Adaptive-weight downsampling
    def __init__(self, ch, group=16) -> None:
        super().__init__()
        
        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(ch, ch, k=1)
        )
        
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))
        
    
    def forward(self, x):
        # bs, ch, 2*h, 2*w => bs, ch, h, w, 4
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)
        
        # bs, 4 * ch, h, w => bs, ch, h, w, 4
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        x = torch.sum(x * att, dim=-1)
        return x
    
######################################## LAWDS end ########################################

######################################## ICCV2025 ESCBlock start ########################################

class C2f_ESC(C2f):
    def __init__(self, c1, c2, n=1, size=None, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ESCBlock(self.c) for _ in range(n))

class Bottleneck_ConvAttn(Bottleneck):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvAttn(c1)
        self.cv2 = ConvAttn(c2)

class C2f_ConvAttn(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_ConvAttn(self.c, self.c, shortcut=shortcut, g=g, e=e) for _ in range(n))

######################################## ICCV2025 ESCBlock end ########################################


######################################## LSKNet start ########################################


class LSKNet(nn.Module):
    """
    根据论文《ALF-YOLO: Enhanced YOLOv8 based on multiscale attention feature fusion for ship detection》
    复现的LSK (Large Selective Kernel)注意力模块。

    该模块通过一个空间选择机制，动态地聚合来自不同大小感受野的特征，
    从而自适应地调整特征提取过程。
    """
    def __init__(self, dim):
        """
        初始化LSK模块。
        :param dim: 输入和输出的通道数。
        """
        super().__init__()
        # ------------------- 步骤 1: 大卷积核分解 -------------------
        # 论文中描述了通过序列卷积来模拟大感受野
        # 分支1: 5x5的深度卷积
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim) # [cite: 391]

        # 分支2: 在分支1的基础上再进行一个7x7的空洞深度卷积
        # 为了获得更大的感受野，这里使用dilation。
        # padding = (kernel_size - 1) * dilation / 2 = (7-1)*3/2 = 9
        self.conv_spatial = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=9, groups=dim, dilation=3) # [cite: 471, 474]
        
        # 两个分支后续的1x1卷积，用于信息融合
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1) # [cite: 474, 476]
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1) # [cite: 492]

        # ------------------- 步骤 2: 空间选择机制 -------------------
        # 用于生成空间注意力图的卷积层 F^(2→N)
        # 输入通道为2（avg_pool和max_pool的拼接），输出通道为N（分支数，这里是2）
        # 论文未指定kernel_size，但7x7是捕获空间上下文的合理选择
        self.conv_squeeze = nn.Conv2d(2, 2, kernel_size=7, padding=3) # [cite: 543]
        
        # Sigmoid激活函数用于生成最终的掩码
        self.sigmoid = nn.Sigmoid() # [cite: 545]

        # ------------------- 步骤 3: 特征融合与输出 -------------------
        # 最终的融合卷积层 F(·)
        self.conv_channel = nn.Conv2d(dim, dim, kernel_size=1) # [cite: 555]

    def forward(self, x):
        """
        前向传播过程
        :param x: 输入特征图，形状为 (B, C, H, W)
        :return: 经过LSK注意力加权后的特征图，形状与输入相同
        """
        B, C, H, W = x.shape

        # --- 1. 大卷积核分解 ---
        # 经过5x5深度卷积的分支
        x_conv0 = self.conv0(x) # [cite: 391]
        
        # 经过5x5 -> 7x7(空洞)深度卷积的分支
        x_conv_spatial = self.conv_spatial(x_conv0) # [cite: 471, 474]

        # 两个分支分别通过1x1卷积得到 U_tilde_1 和 U_tilde_2
        u1 = self.conv1(x_conv0) # [cite: 476]
        u2 = self.conv2(x_conv_spatial) # [cite: 492]

        # --- 2. 空间选择机制 ---
        # 将两个分支的特征拼接在一起，得到 U_tilde
        u_tilde = torch.cat((u1, u2), dim=1) # [cite: 497]

        # 沿着通道维度进行平均池化和最大池化，得到空间描述符 SA_avg 和 SA_max
        # .mean(dim=1, keepdim=True) 会保持维度为 (B, 1, H, W)
        attn_avg = torch.mean(u_tilde, dim=1, keepdim=True) # [cite: 501]
        attn_max, _ = torch.max(u_tilde, dim=1, keepdim=True) # [cite: 502]

        # 拼接两个空间描述符，输入通道变为2
        attn_cat = torch.cat((attn_avg, attn_max), dim=1) # [cite: 543]
        
        # 通过 F^(2→N) 卷积层生成N个空间注意力图
        attn_maps = self.conv_squeeze(attn_cat) # [cite: 543, 544]

        # 通过Sigmoid生成最终的空间选择掩码
        attn_masks = self.sigmoid(attn_maps) # [cite: 545, 547]

        # 将掩码拆分，分别对应两个分支
        mask1 = attn_masks[:, 0:1, :, :]  # 形状 (B, 1, H, W)
        mask2 = attn_masks[:, 1:2, :, :]  # 形状 (B, 1, H, W)

        # --- 3. 特征融合与输出 ---
        # 将掩码与对应的分支特征图相乘，进行加权
        v = self.conv_channel(mask1 * u1 + mask2 * u2) # [cite: 555, 557]

        # 最终输出是原始输入x与注意力特征S(这里是v)的乘积
        output = x * v # [cite: 558, 560]
        
        return output

######################################## LSKNet end ########################################


######################################## SPPF with LSKA start ########################################

class ScaleAdaptiveLSKA(nn.Module):
    """Parallel LSKA branches with a lightweight spatial scale gate."""

    def __init__(self, dim, k_sizes=(7, 11, 23)):
        super().__init__()
        if isinstance(k_sizes, int):
            k_sizes = (k_sizes,)
        self.k_sizes = tuple(int(k) for k in k_sizes)
        self.branches = nn.ModuleList(LSKA(dim, k_size=k) for k in self.k_sizes)
        self.gate = nn.Conv2d(2, len(self.k_sizes), 3, 1, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        avg_attn = torch.mean(x, dim=1, keepdim=True)
        max_attn, _ = torch.max(x, dim=1, keepdim=True)
        weights = self.gate(torch.cat((avg_attn, max_attn), dim=1)).softmax(dim=1)

        out = None
        for i, branch in enumerate(self.branches):
            branch_out = branch(x) * weights[:, i:i + 1]
            out = branch_out if out is None else out + branch_out
        return self.proj(out)


class SPPF_LSKA(nn.Module):
    """SPPF with multi-scale adaptive large separable kernel attention."""

    def __init__(self, c1, c2, k=5, lska_k=(7, 11, 23)):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = ScaleAdaptiveLSKA(c_ * 4, k_sizes=lska_k)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

######################################## SPPF with LSKA end ########################################

class MetaStripMixer(nn.Module):
    """Static strip-aware token mixer for MetaFormer-style neck and context blocks."""

    def __init__(self, dim, band_kernel=7, pool_size=3, drop=0.0):
        super().__init__()
        band_kernel = band_kernel if band_kernel % 2 else band_kernel + 1
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
        self.dw3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.dw_h = nn.Conv2d(dim, dim, (1, band_kernel), 1, (0, band_kernel // 2), groups=dim, bias=False)
        self.dw_v = nn.Conv2d(dim, dim, (band_kernel, 1), 1, (band_kernel // 2, 0), groups=dim, bias=False)
        self.branch_scale = nn.Parameter(torch.ones(4))
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        scales = self.branch_scale.softmax(0)
        pooled = self.pool(x) - x
        y = (
            scales[0] * self.dw3(x)
            + scales[1] * self.dw_h(x)
            + scales[2] * self.dw_v(x)
            + scales[3] * pooled
        )
        return self.proj(y).permute(0, 2, 3, 1)


class MetaSPPF_LSKA_CGLU(nn.Module):
    """SPPF enhanced with LSKA and a MetaFormer-CGLU refinement block."""

    def __init__(self, c1, c2, k=5, strip_k=7):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.refine = MetaFormerCGLUBlock(dim=c2, token_mixer=partial(MetaStripMixer, band_kernel=strip_k))

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        x = self.cv2(self.lska(torch.cat((x, y1, y2, y3), 1)))
        return self.refine(x)


class C2f_MetaStripCGLU(C2f):
    """MetaFormer-inspired C2f with strip-aware token mixing and CGLU channel mixing."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, strip_k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        mixer = partial(MetaStripMixer, band_kernel=strip_k)
        self.m = nn.ModuleList(MetaFormerCGLUBlock(dim=self.c, token_mixer=mixer) for _ in range(n))



######################################## C3 C2f DySnakeConv start ########################################

class Bottleneck_DySnakeConv(Bottleneck):
    """Standard bottleneck with DySnakeConv."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, groups, kernels, expand
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = DySnakeConv(c_, c2, k[1])
        self.cv3 = Conv(c2 * 3, c2, k=1)


    def forward(self, x):
        """'forward()' applies the YOLOv5 FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
    
class C3_DySnakeConv(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Bottleneck_DySnakeConv(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

class C2f_DySnakeConv(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DySnakeConv(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

######################################## C3 C2f DySnakeConv end ########################################


######################################## DyHead begin ########################################
try:
    from mmcv.cnn import build_activation_layer, build_norm_layer
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d
    from mmengine.model import constant_init, normal_init
except ImportError:
    pass

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class h_swish(nn.Module):
    def __init__(self, inplace=False):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class DyReLU(nn.Module):
    def __init__(self, inp, reduction=4, lambda_a=1.0, K2=True, use_bias=True, use_spatial=False,
                 init_a=[1.0, 0.0], init_b=[0.0, 0.0]):
        super(DyReLU, self).__init__()
        self.oup = inp
        self.lambda_a = lambda_a * 2
        self.K2 = K2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.use_bias = use_bias
        if K2:
            self.exp = 4 if use_bias else 2
        else:
            self.exp = 2 if use_bias else 1
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        if reduction == 4:
            squeeze = inp // reduction
        else:
            squeeze = _make_divisible(inp // reduction, 4)
        # print('reduction: {}, squeeze: {}/{}'.format(reduction, inp, squeeze))
        # print('init_a: {}, init_b: {}'.format(self.init_a, self.init_b))

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(inplace=True),
            nn.Linear(squeeze, self.oup * self.exp),
            h_sigmoid()
        )
        if use_spatial:
            self.spa = nn.Sequential(
                nn.Conv2d(inp, 1, kernel_size=1),
                nn.BatchNorm2d(1),
            )
        else:
            self.spa = None

    def forward(self, x):
        if isinstance(x, list):
            x_in = x[0]
            x_out = x[1]
        else:
            x_in = x
            x_out = x
        b, c, h, w = x_in.size()
        y = self.avg_pool(x_in).view(b, c)
        y = self.fc(y).view(b, self.oup * self.exp, 1, 1)
        if self.exp == 4:
            a1, b1, a2, b2 = torch.split(y, self.oup, dim=1)
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]

            b1 = b1 - 0.5 + self.init_b[0]
            b2 = b2 - 0.5 + self.init_b[1]
            out = torch.max(x_out * a1 + b1, x_out * a2 + b2)
        elif self.exp == 2:
            if self.use_bias:  # bias but not PL
                a1, b1 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                b1 = b1 - 0.5 + self.init_b[0]
                out = x_out * a1 + b1

            else:
                a1, a2 = torch.split(y, self.oup, dim=1)
                a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
                a2 = (a2 - 0.5) * self.lambda_a + self.init_a[1]
                out = torch.max(x_out * a1, x_out * a2)

        elif self.exp == 1:
            a1 = y
            a1 = (a1 - 0.5) * self.lambda_a + self.init_a[0]  # 1.0
            out = x_out * a1

        if self.spa:
            ys = self.spa(x_in).view(b, -1)
            ys = F.softmax(ys, dim=1).view(b, 1, h, w) * h * w
            ys = F.hardtanh(ys, 0, 3, inplace=True)/3
            out = out * ys

        return out

class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x


class DyHeadBlock(nn.Module):
    """DyHead Block with three types of attention.
    HSigmoid arguments in default act_cfg follow official code, not paper.
    https://github.com/microsoft/DynamicHead/blob/master/dyhead/dyrelu.py
    """

    def __init__(self,
                 in_channels,
                 norm_type='GN',
                 zero_init_offset=True,
                 act_cfg=dict(type='HSigmoid', bias=3.0, divisor=6.0)):
        super().__init__()
        self.zero_init_offset = zero_init_offset
        # (offset_x, offset_y, mask) * kernel_size_y * kernel_size_x
        self.offset_and_mask_dim = 3 * 3 * 3
        self.offset_dim = 2 * 3 * 3

        if norm_type == 'GN':
            norm_dict = dict(type='GN', num_groups=16, requires_grad=True)
        elif norm_type == 'BN':
            norm_dict = dict(type='BN', requires_grad=True)
        
        self.spatial_conv_high = DyDCNv2(in_channels, in_channels, norm_cfg=norm_dict)
        self.spatial_conv_mid = DyDCNv2(in_channels, in_channels)
        self.spatial_conv_low = DyDCNv2(in_channels, in_channels, stride=2)
        self.spatial_conv_offset = nn.Conv2d(
            in_channels, self.offset_and_mask_dim, 3, padding=1)
        self.scale_attn_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, 1, 1),
            nn.ReLU(inplace=True), build_activation_layer(act_cfg))
        self.task_attn_module = DyReLU(in_channels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, 0, 0.01)
        if self.zero_init_offset:
            constant_init(self.spatial_conv_offset, 0)

    def forward(self, x):
        """Forward function."""
        outs = []
        for level in range(len(x)):
            # calculate offset and mask of DCNv2 from middle-level feature
            offset_and_mask = self.spatial_conv_offset(x[level])
            offset = offset_and_mask[:, :self.offset_dim, :, :]
            mask = offset_and_mask[:, self.offset_dim:, :, :].sigmoid()

            mid_feat = self.spatial_conv_mid(x[level], offset, mask)
            sum_feat = mid_feat * self.scale_attn_module(mid_feat)
            summed_levels = 1
            if level > 0:
                low_feat = self.spatial_conv_low(x[level - 1], offset, mask)
                sum_feat += low_feat * self.scale_attn_module(low_feat)
                summed_levels += 1
            if level < len(x) - 1:
                # this upsample order is weird, but faster than natural order
                # https://github.com/microsoft/DynamicHead/issues/25
                high_feat = F.interpolate(
                    self.spatial_conv_high(x[level + 1], offset, mask),
                    size=x[level].shape[-2:],
                    mode='bilinear',
                    align_corners=True)
                sum_feat += high_feat * self.scale_attn_module(high_feat)
                summed_levels += 1
            outs.append(self.task_attn_module(sum_feat / summed_levels))

        return outs

######################################## Edge information enhancement module start ########################################

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]

class EIEStem(nn.Module):
    def __init__(self, inc, hidc, ouc) -> None:
        super().__init__()
        
        self.conv1 = Conv(inc, hidc, 3, 2)
        self.sobel_branch = SobelConv(hidc)
        self.pool_branch = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)
        )
        self.conv2 = Conv(hidc * 2, hidc, 3, 2)
        self.conv3 = Conv(hidc, ouc, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.sobel_branch(x), self.pool_branch(x)], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class EIEM(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Conv(inc, inc, 3)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Conv(inc, ouc, 1)
    
    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x = self.conv2(x_feature + x)
        return x

class C3_EIEM(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EIEM(c_, c_) for _ in range(n)))

class C2f_EIEM(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EIEM(self.c, self.c) for _ in range(n))


######################################## Edge information enhancement module end ########################################


######################################## SMAFormer start ########################################

class Modulator(nn.Module):
    def __init__(self, in_ch, out_ch, with_pos=True):
        super(Modulator, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.rate = [1, 6, 12, 18]
        self.with_pos = with_pos
        self.patch_size = 2
        self.bias = nn.Parameter(torch.zeros(1, out_ch, 1, 1))

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.CA_fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // 16, in_ch, bias=False),
            nn.Sigmoid(),
        )

        # Pixel Attention
        self.PA_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)
        self.PA_bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention
        self.SA_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=rate, dilation=rate),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_ch)
            ) for rate in self.rate
        ])
        self.SA_out_conv = nn.Conv2d(len(self.rate) * out_ch, out_ch, 1)

        self.output_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_ch)
        self._init_weights()

        self.pj_conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.patch_size + 1,
                         stride=self.patch_size, padding=self.patch_size // 2)
        self.pos_conv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch, bias=True)
        self.layernorm = nn.LayerNorm(self.out_ch, eps=1e-6)

    def forward(self, x):
        res = x
        pa = self.PA(x)
        ca = self.CA(x)

        # Softmax(PA @ CA)
        pa_ca = torch.softmax(pa @ ca, dim=-1)

        # Spatial Attention
        sa = self.SA(x)

        # (Softmax(PA @ CA)) @ SA
        out = pa_ca @ sa
        out = self.norm(self.output_conv(out))
        out = out + self.bias
        synergistic_attn = out + res
        return synergistic_attn

    # def forward(self, x):
    #     pa_out = self.pa(x)
    #     ca_out = self.ca(x)
    #     sa_out = self.sa(x)
    #     # Concatenate along channel dimension
    #     combined_out = torch.cat([pa_out, ca_out, sa_out], dim=1)
    #
    #     return self.norm(self.output_conv(combined_out))



    def PE(self, x):
        proj = self.pj_conv(x)

        if self.with_pos:
            pos = proj * self.sigmoid(self.pos_conv(proj))

        pos = pos.flatten(2).transpose(1, 2)  # BCHW -> BNC
        embedded_pos = self.layernorm(pos)

        return embedded_pos

    def PA(self, x):
        attn = self.PA_conv(x)
        attn = self.PA_bn(attn)
        attn = self.sigmoid(attn)
        return x * attn

    def CA(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.CA_fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def SA(self, x):
        sa_outs = [block(x) for block in self.SA_blocks]
        sa_out = torch.cat(sa_outs, dim=1)
        sa_out = self.SA_out_conv(sa_out)
        return sa_out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class SMA(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(SMA, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_size, num_heads=num_heads, dropout=dropout)
        self.combined_modulator = Modulator(feature_size, feature_size)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, value, key, query):
        MSA = self.attention(query, key, value)[0]

        # 将输出转换为适合AttentionBlock的输入格式
        batch_size, seq_len, feature_size = MSA.shape
        MSA = MSA.permute(0, 2, 1).view(batch_size, feature_size, int(seq_len**0.5), int(seq_len**0.5))
        # 通过CombinedModulator进行multi-attn fusion
        synergistic_attn = self.combined_modulator.forward(MSA)


        # 将输出转换回 (batch_size, seq_len, feature_size) 格式
        x = synergistic_attn.view(batch_size, feature_size, -1).permute(0, 2, 1)

        return x

class E_MLP(nn.Module):
    def __init__(self, feature_size, forward_expansion, dropout):
        super(E_MLP, self).__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_size, forward_expansion * feature_size),
            nn.GELU(),
            nn.Linear(forward_expansion * feature_size, feature_size)
        )
        self.linear1 = nn.Linear(feature_size, forward_expansion * feature_size)
        self.act = nn.GELU()
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(in_channels=forward_expansion * feature_size, out_channels=forward_expansion * feature_size, kernel_size=3, padding=1, groups=1)

        # pixelwise convolution
        self.pixelwise_conv = nn.Conv2d(in_channels=forward_expansion * feature_size, out_channels=forward_expansion * feature_size, kernel_size=3, padding=1)

        self.linear2 = nn.Linear(forward_expansion * feature_size, feature_size)

    def forward(self, x):
        b, hw, c = x.size()
        feature_size = int(math.sqrt(hw))

        x = self.linear1(x)
        x = self.act(x)
        x = rearrange(x, 'b (h w) (c) -> b c h w', h=feature_size, w=feature_size)
        x = self.depthwise_conv(x)
        x = self.pixelwise_conv(x)
        x = rearrange(x, 'b c h w -> b (h w) (c)', h=feature_size, w=feature_size)
        out = self.linear2(x)

        return out

class SMAFormerBlock(nn.Module):
    def __init__(self, ch_out, heads=8, dropout=0.1, forward_expansion=2):
        super(SMAFormerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        self.norm2 = nn.LayerNorm(ch_out)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = E_MLP(ch_out, forward_expansion, dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        value, key, query, res = x, x, x, x
        attention = self.synergistic_multi_attention(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query)
        out = self.dropout(self.norm2(feed_forward + query))
        return out.permute(0, 2, 1).reshape((b, c, h, w))

class SMAFormerBlock_CGLU(nn.Module):
    def __init__(self, ch_out, heads=8, dropout=0.1, forward_expansion=2):
        super(SMAFormerBlock_CGLU, self).__init__()
        self.norm1 = nn.LayerNorm(ch_out)
        # self.norm2 = nn.LayerNorm(ch_out)
        self.norm2 = LayerNorm2d(ch_out)
        self.synergistic_multi_attention = SMA(ch_out, heads, dropout)
        self.e_mlp = ConvolutionalGLU(ch_out, forward_expansion, drop=dropout)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.flatten(2).permute(0, 2, 1)
        value, key, query, res = x, x, x, x
        attention = self.synergistic_multi_attention(query, key, value)
        query = self.dropout(self.norm1(attention + res))
        feed_forward = self.e_mlp(query.permute(0, 2, 1).reshape((b, c, h, w)))
        out = self.dropout(self.norm2(feed_forward))
        return out

class C2f_SMAFB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SMAFormerBlock(self.c) for _ in range(n))
        
class C2f_SMAFB_CGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SMAFormerBlock_CGLU(self.c) for _ in range(n))

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
    
    # def forward(self, x):
    #     x, v = self.fc1(x).chunk(2, dim=1)
    #     x = self.dwconv(x) * v
    #     x = self.drop(x)
    #     x = self.fc2(x)
    #     x = self.drop(x)
    #     return x

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x
######################################## SMAFormer end ########################################



######################################## MutilScaleEdgeInformationEnhance start ########################################

# 1.使用 nn.AvgPool2d 对输入特征图进行平滑操作，提取其低频信息。
# 2.将原始输入特征图与平滑后的特征图进行相减，得到增强的边缘信息（高频信息）。
# 3.用卷积操作进一步处理增强的边缘信息。
# 4.将处理后的边缘信息与原始输入特征图相加，以形成增强后的输出。
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class MutilScaleEdgeInformationEnhance(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)] 
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))

class MutilScaleEdgeInformationSelect(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.dsm = DualDomainSelectionMechanism(inc * 2)
        self.final_conv = Conv(inc * 2, inc)
    
    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))

class CSP_MutilScaleEdgeInformationEnhance(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MutilScaleEdgeInformationEnhance(self.c, [3, 6, 9, 12]) for _ in range(n))

class CSP_MutilScaleEdgeInformationSelect(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MutilScaleEdgeInformationSelect(self.c, [3, 6, 9, 12]) for _ in range(n))        
        
######################################## GlobalEdgeInformationTransfer start ########################################

class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]

class MutilScaleEdgeInfoGenetator(nn.Module):
    def __init__(self, inc, oucs) -> None:
        super().__init__()
        
        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)
    
    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        return outputs

class ConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k = 1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)
    
    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x

######################################## GlobalEdgeInformationTransfer end ########################################


######################################## MetaFormer Baselines for Vision TPAMI2024 start ########################################

class C2f_IdentityFormer(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerBlock(
                dim=self.c, token_mixer=nn.Identity, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

class C2f_RandomMixing(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, num_tokens=196, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerBlock(
                dim=self.c, token_mixer=partial(RandomMixing, num_tokens=num_tokens), norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

class C2f_PoolingFormer(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerBlock(
                dim=self.c, token_mixer=Pooling, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))
        
class C2f_ConvFormer(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerBlock(
                dim=self.c, token_mixer=SepConv
            ) for _ in range(n))
        
class C2f_CaFormer(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerBlock(
                dim=self.c, token_mixer=MF_Attention
            ) for _ in range(n))

class C2f_IdentityFormerCGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerCGLUBlock(
                dim=self.c, token_mixer=nn.Identity, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

class C2f_RandomMixingCGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, num_tokens=196, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerCGLUBlock(
                dim=self.c, token_mixer=partial(RandomMixing, num_tokens=num_tokens), norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))

class C2f_PoolingFormerCGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerCGLUBlock(
                dim=self.c, token_mixer=Pooling, norm_layer=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False)
            ) for _ in range(n))
        
class C2f_ConvFormerCGLU(C2f):
    """C2f block with ConvFormer-style token mixing and convolutional gated linear units."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerCGLUBlock(
                dim=self.c, token_mixer=SepConv
            ) for _ in range(n))
        
class C2f_CaFormerCGLU(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MetaFormerCGLUBlock(
                dim=self.c, token_mixer=MF_Attention
            ) for _ in range(n))

######################################## MetaFormer Baselines for Vision TPAMI2024 end ########################################


######################################## Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection start ########################################

class PSConv(nn.Module):  
    ''' Pinwheel-shaped Convolution using the Asymmetric Padding method. '''
    
    def __init__(self, c1, c2, k, s):
        super().__init__()

        # self.k = k
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))

class APBottleneck(nn.Module):
    """Asymmetric Padding bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        p = [(2,0,2,0),(0,2,0,2),(0,2,2,0),(2,0,0,2)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        # self.cv1 = nn.ModuleList([nn.Conv2d(c1, c_, k[0], stride=1, padding= p[g], bias=False) for g in range(4)])
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # y = self.pad[g](x) for g in range(4)
        return x + self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1))) if self.add else self.cv2((torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1)))

class C2f_AP(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(APBottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=e) for _ in range(n))

######################################## Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection end ########################################

######################################## StartNet end ########################################

class Star_Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = Conv(dim, dim, 7, g=dim, act=False)
        self.f1 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.f2 = nn.Conv2d(dim, mlp_ratio * dim, 1)
        self.g = Conv(mlp_ratio * dim, dim, 1, act=False)
        self.dwconv2 = nn.Conv2d(dim, dim, 7, 1, (7 - 1) // 2, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class Star_Block_CAA(Star_Block):
    def __init__(self, dim, mlp_ratio=3, drop_path=0):
        super().__init__(dim, mlp_ratio, drop_path)
        
        self.attention = CAA(mlp_ratio * dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(self.attention(x)))
        x = input + self.drop_path(x)
        return x

class C3_Star(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block(c_) for _ in range(n)))

class C2f_Star(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block(self.c) for _ in range(n))

class C3_Star_CAA(C3):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Star_Block_CAA(c_) for _ in range(n)))

class C2f_Star_CAA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Star_Block_CAA(self.c) for _ in range(n))

class ECALite(nn.Module):
    """Low-cost channel recalibration used by SGR block."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.act(y)
        return x * y


class RepDWConv(nn.Module):
    """Depthwise re-parameterizable conv branch used in SGR."""

    def __init__(self, channels, k=3, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.channels = channels
        self.k = k
        padding = k // 2
        if deploy:
            self.rbr_reparam = nn.Conv2d(channels, channels, k, 1, padding, groups=channels, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(channels, channels, k, 1, padding, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(channels, channels, 1, 1, 0, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            self.rbr_identity = nn.BatchNorm2d(channels)

    def forward(self, x):
        if self.deploy:
            return self.rbr_reparam(x)
        return self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return w * t, beta - mean * gamma / std

    @staticmethod
    def _pad_1x1_to_kxk(kernel, k):
        if kernel is None:
            return 0
        if kernel.size(2) == k and kernel.size(3) == k:
            return kernel
        pad = (k - 1) // 2
        return F.pad(kernel, [pad, pad, pad, pad])

    def _fuse_identity_bn(self):
        if not hasattr(self, "rbr_identity"):
            return 0, 0
        k = self.k
        kernel = torch.zeros((self.channels, 1, k, k), device=self.rbr_identity.weight.device)
        kernel[:, 0, k // 2, k // 2] = 1.0
        bn = self.rbr_identity
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = torch.sqrt(var + eps)
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - mean * gamma / std

    def get_equivalent_kernel_bias(self):
        if self.deploy:
            return self.rbr_reparam.weight, self.rbr_reparam.bias
        k3, b3 = self._fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        kid, bid = self._fuse_identity_bn()
        k1 = self._pad_1x1_to_kxk(k1, self.k)
        kernel = k3 + k1 + kid
        bias = b3 + b1 + bid
        return kernel, bias

    def switch_to_deploy(self):
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.channels,
            self.channels,
            self.k,
            1,
            self.k // 2,
            groups=self.channels,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.deploy = True


class SGR_Block(nn.Module):
    """Star-Gated Reparam block for lightweight neck replacement."""

    def __init__(self, dim, mlp_ratio=2.0, drop_path=0.0, deploy=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.value = RepDWConv(dim, k=3, deploy=deploy)
        self.gate = RepDWConv(dim, k=5, deploy=deploy)
        self.pw1 = Conv(dim, hidden, 1, 1)
        self.pw2 = Conv(hidden, dim, 1, 1, act=False)
        self.eca = ECALite(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.value(x) * torch.sigmoid(self.gate(x))
        x = self.pw2(self.pw1(x))
        x = self.eca(x)
        return identity + self.drop_path(x)

    def switch_to_deploy(self):
        self.value.switch_to_deploy()
        self.gate.switch_to_deploy()


class C2f_SGR(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(SGR_Block(self.c) for _ in range(n))

class SGRNeck(nn.Module):
    """
    Standalone neck cell (non-C2f):
    1) channel align
    2) star-gated depthwise interaction (local * context gate)
    3) stacked SGR blocks at reduced width
    4) project back to target channels
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        c_ = max(int(c2 * e), 16)
        self.shortcut = shortcut and (c1 == c2)
        self.cv_in = Conv(c1, c_, 1, 1)
        self.local = Conv(c_, c_, 3, 1, g=c_, act=False)
        self.gate = RepDWConv(c_, k=5, deploy=False)
        self.blocks = nn.Sequential(*(SGR_Block(c_) for _ in range(n)))
        self.cv_out = Conv(c_, c2, 1, 1)

    def forward(self, x):
        identity = x
        x = self.cv_in(x)
        x = self.local(x) * torch.sigmoid(self.gate(x)) + x
        x = self.blocks(x)
        x = self.cv_out(x)
        if self.shortcut:
            x = x + identity
        return x

######################################## StartNet end ########################################

class DyDCNv2(nn.Module):
    """ModulatedDeformConv2d with normalization layer used in DyHead.
    This module cannot be configured with `conv_cfg=dict(type='DCNv2')`
    because DyHead calculates offset and mask from middle-level feature.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int | tuple[int], optional): Stride of the convolution.
            Default: 1.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: dict(type='GN', num_groups=16, requires_grad=True).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)):
        super().__init__()
        self.with_norm = norm_cfg is not None
        bias = not self.with_norm
        self.conv = ModulatedDeformConv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=bias)
        if self.with_norm:
            self.norm = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x, offset, mask):
        """Forward function."""
        x = self.conv(x.contiguous(), offset, mask)
        if self.with_norm:
            x = self.norm(x)
        return x
    
######################################## Focus Diffusion Pyramid Network  ########################################
class FocusFeature(nn.Module):
    def __init__(self, inc, kernel_sizes=(5, 7, 9, 11), e=0.5) -> None:
        super().__init__()
        hidc = int(inc[1] * e)
        
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(inc[0], hidc, 1)
        )
        self.conv2 = Conv(inc[1], hidc, 1) if e != 1 else nn.Identity()
        self.conv3 = ADown(inc[2], hidc)
        
        
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc * 3, hidc * 3, kernel_size=k, padding=autopad(k), groups=hidc * 3) for k in kernel_sizes)
        self.pw_conv = Conv(hidc * 3, hidc * 3)
    
    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)
        feature = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv], dim=0), dim=0)
        feature = self.pw_conv(feature)
        
        x = x + feature
        return x

######################################## Focus Diffusion Pyramid Network  ########################################

class StripContextGate(nn.Module):
    """Lightweight strip-context gate for low-cost long-range aggregation."""

    def __init__(self, c, k=11):
        super().__init__()
        self.h_conv = nn.Conv1d(c, c, k, padding=k // 2, groups=c, bias=False)
        self.w_conv = nn.Conv1d(c, c, k, padding=k // 2, groups=c, bias=False)
        self.proj = nn.Conv2d(c, c, 1, 1, bias=True)

    def forward(self, x):
        h_ctx = self.h_conv(x.mean(dim=3, keepdim=False)).unsqueeze(-1)
        w_ctx = self.w_conv(x.mean(dim=2, keepdim=False)).unsqueeze(-2)
        return torch.sigmoid(self.proj(h_ctx + w_ctx))


class HybridKernelStateBlock(nn.Module):
    """Hybrid local-global mixer for neck features."""

    def __init__(self, c, shortcut=True, lk=7, strip_k=11, gate_ratio=0.25):
        super().__init__()
        hidden = max(16, int(c * gate_ratio))
        pad = lk // 2
        self.shortcut = shortcut
        self.pre = Conv(c, c, 1, 1)
        self.dw3 = nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False)
        self.dw5 = nn.Conv2d(c, c, 5, 1, 2, groups=c, bias=False)
        self.dwlk = nn.Conv2d(c, c, lk, 1, pad, groups=c, bias=False)
        self.local_fuse = Conv(3 * c, c, 1, 1)
        self.context_gate = StripContextGate(c, strip_k)
        self.path_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * c, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.out = Conv(c, c, 1, 1)

    def forward(self, x):
        identity = x
        x = self.pre(x)
        local = self.local_fuse(torch.cat((self.dw3(x), self.dw5(x), self.dwlk(x)), 1))
        context = x * self.context_gate(x)
        g_local, g_context = self.path_gate(x).chunk(2, 1)
        out = self.out(g_local * local + g_context * context)
        return identity + out if self.shortcut else out


class C2f_HKSM(nn.Module):
    """
    C2f with a Hybrid Kernel-State Mixer.

    Design motivation:
    - Preserve the efficient partial aggregation path of C2f for YOLO necks.
    - Use poly-kernel depthwise mixing for stronger local multi-scale modeling.
    - Add strip-context gating to inject low-cost long-range context.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, lk=7, strip_k=11):
        super().__init__()
        # Ultralytics parse_model follows the common C2f positional order:
        # [c2, shortcut, g, e]. In custom YAML we often want [c2, shortcut, e, lk, strip_k].
        # If a YAML entry such as [512, False, 0.5, 7, 11] is passed positionally, it becomes
        # g=0.5, e=7, lk=11, which explodes hidden channels. Detect and fix that case here.
        if e > 2 and 0 < g <= 1:
            strip_k = lk
            lk = int(e)
            e = float(g)
            g = 1
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(HybridKernelStateBlock(self.c, shortcut, lk, strip_k) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

######################################## FeaturePyramidSharedConv Module start ########################################

class FeaturePyramidSharedConv(nn.Module):
    def __init__(self, c1, c2, dilations=[1, 3, 5]) -> None:
        super().__init__()

        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (1 + len(dilations)), c2, 1, 1)
        self.share_conv = nn.Conv2d(in_channels=c_, out_channels=c_, kernel_size=3, stride=1, padding=1, bias=False)
        self.dilations = dilations
    
    def forward(self, x):
        y = [self.cv1(x)]
        for dilation in self.dilations:
            y.append(F.conv2d(y[-1], weight=self.share_conv.weight, bias=None, dilation=dilation, padding=(dilation * (3 - 1) + 1) // 2))
        return self.cv2(torch.cat(y, 1))
        
######################################## FeaturePyramidSharedConv Module end ########################################

class DCT_Pooling(nn.Module):
    # 频率域池化层：利用离散余弦变换捕捉全局结构
    def __init__(self, channels):
        super().__init__()
        self.register_buffer('weight', self._get_dct_filter(channels))

    def _get_dct_filter(self, c):
        # 生成 DCT 基向量，用于频域加权
        dct_filter = torch.zeros(c, c, 1, 1)
        # 这里简化为频率选择掩模，实际可实现完整的 2D-DCT 变换
        for i in range(c):
            dct_filter[i, i, 0, 0] = 1.0 / (i + 1) # 低频优先策略
        return dct_filter

    def forward(self, x):
        # 模拟频域特征提取
        _, _, h, w = x.shape
        x_dtype = x.dtype
        x_fft = x.float() if x_dtype in (torch.float16, torch.bfloat16) else x
        # 使用全局平均值作为 DC 分量，并叠加频域权重
        res = torch.fft.rfft2(x_fft, norm='backward') # 快速傅里叶变换
        res = torch.fft.irfft2(res, s=(h, w), norm='backward')
        if res.dtype != x_dtype:
            res = res.to(x_dtype)
        return x * torch.sigmoid(res)

class Spectra_SPPF(nn.Module):
    # 高级创新：频谱感知 SPPF
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        
        # 替换池化为多尺度频域滤波器
        self.m1 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.f2 = DCT_Pooling(c_) # 频域特征增强分支
        self.m3 = nn.MaxPool2d(kernel_size=k*2-1, stride=1, padding=(k*2-1) // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m1(x)
        y2 = self.f2(y1) # 在空间池化基础上注入频域全局上下文
        y3 = self.m3(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))

class BiRoutingBlock(nn.Module):
    # 动态路由模块：只在相关联的区域间进行特征传递
    def __init__(self, dim, num_heads=8, topk=4):
        super().__init__()
        self.topk = topk
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        # 路由映射：决定哪些区域该跟哪些区域“对话”
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(8), # 划分 8x8 区域
            nn.Conv2d(dim, dim, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        # 1. 生成区域级路由权重
        route_weights = self.router(x) # [B, C, 8, 8]
        
        # 2. 选取 Top-K 最相关的区域 (Sparse Routing)
        # 这一步在代码中通过掩码实现，只允许高贡献区域参与计算
        _, indices = torch.topk(route_weights, self.topk, dim=1)
        mask = torch.zeros_like(route_weights).scatter_(1, indices, 1.0)
        
        # 3. 执行动态加权特征聚合
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)
        
        # 将路由掩码上采样回原图尺寸作用于特征
        mask_up = nn.functional.interpolate(mask, size=(h, w), mode='nearest')
        out = (q * mask_up) + v # 简化演示：只在路由区域增强特征
        
        return self.proj(out)

class DSR_C2f(nn.Module):
    # 替换 C2f：基于动态稀疏路由的颈部节点
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([BiRoutingBlock(self.c) for _ in range(n)]) # 路由替换 Bottleneck

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class SpectralChannelGate(nn.Module):
    """Lightweight spectral recalibration that preserves the SPPF inductive bias."""

    def __init__(self, c, reduction=4):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.fc1 = nn.Conv2d(c, hidden, 1, 1, bias=True)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, c, 1, 1, bias=True)

    def forward(self, x):
        x_dtype = x.dtype
        x_freq = x.float() if x_dtype in (torch.float16, torch.bfloat16) else x
        amp = torch.abs(torch.fft.rfft2(x_freq, norm="ortho"))
        desc = amp.mean(dim=(-2, -1), keepdim=True)
        gate = torch.sigmoid(self.fc2(self.act(self.fc1(desc.to(x.dtype)))))
        return gate


class SpectraLite_SPPF(nn.Module):
    """
    Detection-friendly SPPF variant.

    It keeps the original progressive max-pooling path and only adds a mild spectral
    channel gate to recalibrate pooled features, which is much safer than replacing
    the pooling path with a hard frequency-domain branch.
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.spectral_gate = SpectralChannelGate(c_)

    def forward(self, x):
        x = self.cv1(x)
        gate = self.spectral_gate(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1 * gate, y2 * gate, y3 * gate), 1))


class SpatialRoutingGate(nn.Module):
    """Soft spatial routing gate used to modulate local and context paths."""

    def __init__(self, c, pool_size=8, reduction=4):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(c, hidden, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, 1, bias=True),
        )

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.router(x), size=x.shape[-2:], mode="nearest"))


class RoutingResidualBlock(nn.Module):
    """
    Residual local-global mixer for detection necks.

    Compared with the original DSR block, this version keeps convolutional local
    modeling, uses soft routing over spatial positions, and preserves a residual path.
    """

    def __init__(self, c, shortcut=True, mlp_ratio=1.5):
        super().__init__()
        hidden = max(int(c * mlp_ratio), c)
        self.shortcut = shortcut
        self.pre = Conv(c, c, 1, 1)
        self.dw3 = Conv(c, c, 3, 1, g=c)
        self.dw5 = Conv(c, c, 5, 1, g=c)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.route = SpatialRoutingGate(c)
        self.mix = Conv(2 * c, c, 1, 1)
        self.ffn = nn.Sequential(
            Conv(c, hidden, 1, 1),
            Conv(hidden, c, 1, 1, act=False),
        )

    def forward(self, x):
        identity = x
        x = self.pre(x)
        route = self.route(x)
        local = self.dw3(x) + self.dw5(x)
        context = x * self.context(x)
        fused = self.mix(torch.cat((local * route, context * (1 - route)), 1))
        out = self.ffn(fused)
        return identity + out if self.shortcut else out


class RDS_C2f(C2f):
    """
    C2f with Residual Dynamic Spatial routing blocks.

    This keeps the stable C2f aggregation pattern while replacing the internal
    bottleneck with a softer routing-aware block designed for dense detection.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RoutingResidualBlock(self.c, shortcut=True) for _ in range(n))


class EdgePriorExtractor(nn.Module):
    """Fixed Sobel edge prior followed by lightweight channel projection."""

    def __init__(self, c):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32)
        self.register_buffer("weight_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("weight_y", sobel_y.view(1, 1, 3, 3))
        self.proj = Conv(1, c, 1, 1)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        # Keep the Sobel branch on the same device/dtype as the current input path to
        # avoid AMP/half precision mismatches during validation and inference.
        weight_x = self.weight_x.to(device=gray.device, dtype=gray.dtype)
        weight_y = self.weight_y.to(device=gray.device, dtype=gray.dtype)
        edge_x = F.conv2d(gray, weight_x, padding=1)
        edge_y = F.conv2d(gray, weight_y, padding=1)
        edge = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-6)
        return self.proj(edge)


class SpectraEdge_SPPF(nn.Module):
    """
    Stronger SPPF variant with spectral recalibration and explicit edge prior.

    The main path stays close to SPPF, while pooled features are selectively enhanced
    by spectral gates and a learned edge prior to better retain detection boundaries.
    """

    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.spectral_gate = SpectralChannelGate(c_)
        self.edge_prior = EdgePriorExtractor(c_)
        self.edge_gate = nn.Sequential(
            Conv(c_, c_, 3, 1, g=c_),
            nn.Conv2d(c_, c_, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.cv1(x)
        spec_gate = self.spectral_gate(x)
        edge = self.edge_prior(x)
        edge_gate = self.edge_gate(edge)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        y1 = y1 * spec_gate + edge
        y2 = y2 * spec_gate + edge * edge_gate
        y3 = y3 * spec_gate + edge * (1.0 - edge_gate)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


class DualExpertGate(nn.Module):
    """Generates two spatially-aware expert weights."""

    def __init__(self, c, reduction=4):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 2 * c, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).chunk(2, 1)


class ResidualDualExpertBlock(nn.Module):
    """
    Two-expert neck block:
    - local expert keeps dense convolutional modeling
    - structure expert injects edge/context priors
    - soft gating mixes both experts before a residual FFN
    """

    def __init__(self, c, shortcut=True, mlp_ratio=1.5):
        super().__init__()
        hidden = max(int(c * mlp_ratio), c)
        self.shortcut = shortcut
        self.pre = Conv(c, c, 1, 1)
        self.local_expert = nn.Sequential(
            Conv(c, c, 3, 1, g=c),
            Conv(c, c, 5, 1, g=c),
        )
        self.structure_expert = nn.Sequential(
            EdgePriorExtractor(c),
            Conv(c, c, 3, 1, g=c),
        )
        self.context_gate = SpatialRoutingGate(c)
        self.expert_gate = DualExpertGate(c)
        self.mix = Conv(2 * c, c, 1, 1)
        self.ffn = nn.Sequential(
            Conv(c, hidden, 1, 1),
            Conv(hidden, c, 1, 1, act=False),
        )

    def forward(self, x):
        identity = x
        x = self.pre(x)
        route = self.context_gate(x)
        w_local, w_struct = self.expert_gate(x)
        local = self.local_expert(x)
        struct = self.structure_expert(x)
        fused = self.mix(torch.cat((local * (w_local + route), struct * (w_struct + 1.0 - route)), 1))
        out = self.ffn(fused)
        return identity + out if self.shortcut else out


class RDEA_C2f(C2f):
    """C2f with Residual Dual-Expert Aggregation blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ResidualDualExpertBlock(self.c, shortcut=True) for _ in range(n))


class LSCDStatsGate(nn.Module):
    """Mean-std channel calibration tailored for shared detection heads."""

    def __init__(self, c, reduction=8):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.net = nn.Sequential(
            nn.Conv2d(2 * c, hidden, 1, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1, 1, bias=True),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)
        gate = 2.0 * self.net(torch.cat((mean, std), 1))
        return x * gate


class StarRouteGate(nn.Module):
    """Low-cost spatial routing between detail and semantic branches."""

    def __init__(self, c, reduction=4, pool_size=8):
        super().__init__()
        hidden = max(c // reduction, 16)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Conv2d(c, hidden, 1, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 1, 1, bias=True),
        )

    def forward(self, x):
        return torch.sigmoid(F.interpolate(self.net(x), size=x.shape[-2:], mode="nearest"))


class StarAlignBlock(nn.Module):
    """
    StarNet-style dual-branch mixer for lightweight necks.

    - detail branch: cheap re-parameterized 3x3 depthwise conv for edges and small objects
    - semantic branch: large-kernel depthwise conv followed by StarNet-style multiplicative gating
    - tail stats gate: aligns channel response ranges before LSCD's shared conv head
    """

    def __init__(self, c, shortcut=True, mlp_ratio=2.0, lk=7, reduction=8):
        super().__init__()
        hidden = max(int(c * mlp_ratio), c)
        lk = lk if lk % 2 else lk + 1
        self.shortcut = shortcut
        self.pre = Conv(c, c, 1, 1)
        self.detail = RepDWConv(c, k=3, deploy=False)
        self.semantic = RepDWConv(c, k=lk, deploy=False)
        self.f1 = nn.Conv2d(c, hidden, 1, 1, bias=True)
        self.f2 = nn.Conv2d(c, hidden, 1, 1, bias=True)
        self.act = nn.ReLU6()
        self.star_proj = Conv(hidden, c, 1, 1, act=False)
        self.route = StarRouteGate(c)
        self.align = LSCDStatsGate(c, reduction=reduction)
        self.out = Conv(c, c, 1, 1, act=False)

    def forward(self, x):
        identity = x
        x = self.pre(x)
        detail = self.detail(x)
        semantic = self.semantic(x)
        semantic = self.star_proj(self.act(self.f1(semantic)) * self.f2(semantic))
        route = self.route(x)
        fused = detail * route + semantic * (1.0 - route)
        out = self.out(self.align(fused))
        return identity + out if self.shortcut else out

    def switch_to_deploy(self):
        self.detail.switch_to_deploy()
        self.semantic.switch_to_deploy()


class C2f_StarAlign(C2f):
    """C2f co-designed for StarNet backbones and LSCD shared-conv heads."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, lk=7, reduction=8):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(StarAlignBlock(self.c, shortcut=True, lk=lk, reduction=reduction) for _ in range(n))
        self.tail_align = LSCDStatsGate(c2, reduction=reduction)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.tail_align(self.cv2(torch.cat(y, 1)))

class AINLiteBlock(nn.Module):
    """
    Deployment-friendly AIN variant for detection necks.

    Design choices for front-view vehicle detection:
    - keep only static convolutional branches from AIN
    - retain a native identity path for LSCD-friendly feature stability
    - avoid ISA/GRN/shuffle to reduce real latency overhead
    """

    def __init__(self, c, shortcut=True, band_k=7):
        super().__init__()
        band_k = band_k if band_k % 2 else band_k + 1
        self.shortcut = shortcut
        base = c // 4
        self.splits = (base, base, base, c - 3 * base)
        c_sq, c_h, c_v, _ = self.splits
        self.dw_square = nn.Sequential(
            nn.Conv2d(c_sq, c_sq, 3, 1, 1, groups=c_sq, bias=False),
            nn.BatchNorm2d(c_sq),
        )
        self.dw_h = nn.Sequential(
            nn.Conv2d(c_h, c_h, (1, band_k), 1, (0, band_k // 2), groups=c_h, bias=False),
            nn.BatchNorm2d(c_h),
        )
        self.dw_v = nn.Sequential(
            nn.Conv2d(c_v, c_v, (band_k, 1), 1, (band_k // 2, 0), groups=c_v, bias=False),
            nn.BatchNorm2d(c_v),
        )
        self.mix = Conv(c, c, 1, 1)
        self.proj = Conv(c, c, 1, 1, act=False)

    def forward(self, x):
        identity = x
        x_sq, x_h, x_v, x_id = torch.split(x, self.splits, 1)
        y = torch.cat((self.dw_square(x_sq), self.dw_h(x_h), self.dw_v(x_v), x_id), 1)
        y = self.proj(self.mix(y))
        return identity + y if self.shortcut else y


class C2f_AINLite(C2f):
    """
    AIN-inspired C2f for lightweight vehicle detection.

    It borrows the static strip-convolution idea from AIN-YOLO, but keeps the
    implementation neck-friendly and latency-aware for StarNet + LSCD setups.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, band_k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(AINLiteBlock(self.c, shortcut=True, band_k=band_k) for _ in range(n))

class ECALite(nn.Module):
    """Low-cost channel recalibration used by SGR block."""

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log2(channels) + b) / gamma))
        k = k if k % 2 else k + 1
        k = max(k, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.act(y)
        return x * y


class C2f_AINLite_ECA(C2f):
    """
    AINLite neck block with ECA.

    This is the first recommended attention variant for your deployment-sensitive
    setting because it only adds a tiny channel calibration after static strip
    convolutions.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, band_k=7):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(AINLiteBlock(self.c, shortcut=True, band_k=band_k) for _ in range(n))
        self.eca = ECALite(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.eca(self.cv2(torch.cat(y, 1)))
