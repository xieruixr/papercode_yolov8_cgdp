"""SPECA block used by YOLOv8-CGDP."""

import torch
import torch.nn as nn


class SPC_Module(nn.Module):
    """Spatial pyramid convolution with parallel kernel sizes."""

    def __init__(self, c_in, k_sizes=(3, 5, 7)):
        super().__init__()
        if not k_sizes:
            raise ValueError("k_sizes must contain at least one kernel size.")
        if c_in < len(k_sizes):
            raise ValueError(f"c_in={c_in} must be >= number of branches ({len(k_sizes)}).")

        self.S = len(k_sizes)
        self.c_branch = int(c_in / self.S)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c_in, self.c_branch, kernel_size=k, padding=k // 2, bias=False),
                nn.BatchNorm2d(self.c_branch),
                nn.SiLU(inplace=True),
            )
            for k in k_sizes
        )

    def forward(self, x):
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class CSA_Module(nn.Module):
    """Contextual scale aggregation over the SPC branch output."""

    def __init__(self, c_in, k_sizes=(3, 5, 7)):
        super().__init__()
        self.spc = SPC_Module(c_in, k_sizes)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.spc(x)
        weights = self.gap(features).squeeze(-1).permute(0, 2, 1)
        weights = self.conv(weights).permute(0, 2, 1).unsqueeze(-1)
        return features * self.sigmoid(weights)


class ConvBranch(nn.Module):
    """Lightweight spatial refinement branch with residual gating."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.ReLU(inplace=True),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        identity = x
        refined = self.conv1(x)
        refined = refined + self.conv2(refined)
        refined = self.conv3(refined)
        refined = refined + self.conv4(refined)
        refined = self.conv5(refined)
        refined = refined + self.conv6(refined)
        refined = self.conv7(refined)
        return identity + identity * self.sigmoid_spatial(refined)


class SPECA(nn.Module):
    """Spatial Pyramid Enhanced Context Aggregation block."""

    def __init__(self, c_in, k_sizes=(3, 5, 7)):
        super().__init__()
        self.csa = CSA_Module(c_in, k_sizes=k_sizes)
        c_total = self.csa.spc.S * self.csa.spc.c_branch
        self.conv_branch = ConvBranch(in_features=c_total, hidden_features=c_total, out_features=c_in)

    def forward(self, x):
        return x + self.conv_branch(self.csa(x))


if __name__ == "__main__":
    sample = torch.randn(4, 64, 20, 20)
    block = SPECA(c_in=64, k_sizes=(3, 5, 7, 9))
    output = block(sample)
    assert output.shape == sample.shape
    print("SPECA smoke test passed:", output.shape)
