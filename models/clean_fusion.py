from __future__ import annotations

import math
import random
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from utils.pca import FixedPCATransform
import torch.nn.functional as F


def _bn(channels: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        act=True,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                padding = kernel_size // 2
        layers = [
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            _bn(out_ch),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class GroupMixStyle(nn.Module):
    """A small grouped MixStyle-style augmentation for the HSI branch.

    This is intentionally much lighter than CMGF's HyperGroupMix. It is used
    only during training and has no learnable parameters.
    """

    def __init__(self, groups: int = 4, p: float = 0.25, alpha: float = 0.2, eps: float = 1e-6) -> None:
        super().__init__()
        self.groups = int(groups)
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p <= 0 or random.random() > self.p or x.shape[0] < 2:
            return x
        b, c, _, _ = x.shape
        groups = min(self.groups, c)
        while groups > 1 and c % groups != 0:
            groups -= 1
        cg = c // groups
        xg = x.reshape(b, groups, cg, x.shape[-2], x.shape[-1])
        mu = xg.mean(dim=(-2, -1), keepdim=True)
        var = xg.var(dim=(-2, -1), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        normed = (xg - mu) / sig
        perm = torch.randperm(b, device=x.device)
        beta = torch.distributions.Beta(self.alpha, self.alpha)
        lam = beta.sample((b, 1, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        mu_mix = lam * mu + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]
        return (normed * sig_mix + mu_mix).reshape_as(x)


class SelectiveLargeKernelBlock(nn.Module):
    """Parameter-efficient local/mid/large receptive-field block.

    Three depthwise branches (3x3, asymmetric 7, asymmetric 11) are selected
    per sample by a lightweight softmax gate, then mixed by a pointwise FFN.
    """

    def __init__(self, channels: int, expansion: float = 2.0, dropout: float = 0.05) -> None:
        super().__init__()
        c = int(channels)
        hidden = max(c, int(round(c * expansion)))
        self.pre = _bn(c)
        self.b3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.b7 = nn.Sequential(
            nn.Conv2d(c, c, (1, 7), padding=(0, 3), groups=c, bias=False),
            nn.Conv2d(c, c, (7, 1), padding=(3, 0), groups=c, bias=False),
        )
        self.b11 = nn.Sequential(
            nn.Conv2d(c, c, (1, 11), padding=(0, 5), groups=c, bias=False),
            nn.Conv2d(c, c, (11, 1), padding=(5, 0), groups=c, bias=False),
        )
        gate_hidden = max(16, c // 4)
        self.gate_reduce = nn.Conv2d(c, gate_hidden, 1, bias=True)
        self.gate_expand = nn.Conv2d(gate_hidden, c * 3, 1, bias=True)
        self.spatial_mix = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            _bn(c),
            nn.GELU(),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            _bn(hidden),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, c, 1, bias=False),
            _bn(c),
        )
        self.layer_scale1 = nn.Parameter(torch.full((1, c, 1, 1), 0.5))
        self.layer_scale2 = nn.Parameter(torch.full((1, c, 1, 1), 0.5))

    def forward(self, x: Tensor) -> Tensor:
        z = self.pre(x)
        branches = torch.stack([self.b3(z), self.b7(z), self.b11(z)], dim=1)  # B,3,C,H,W
        context = branches.sum(dim=1).mean(dim=(-2, -1), keepdim=True)
        weights = self.gate_expand(F.gelu(self.gate_reduce(context)))
        b, _, _, _ = weights.shape
        weights = weights.view(b, 3, x.shape[1], 1, 1).softmax(dim=1)
        selected = (branches * weights).sum(dim=1)
        x = x + self.layer_scale1 * self.spatial_mix(selected)
        x = x + self.layer_scale2 * self.ffn(F.gelu(x))
        return F.gelu(x)


class HSIBackbone(nn.Module):
    def __init__(self, num_bands: int, dim: int = 96, dropout: float = 0.05, mixstyle_p: float = 0.25) -> None:
        super().__init__()
        c1, c2, c3 = 48, 64, int(dim)
        self.spectral_projection = nn.Sequential(
            nn.Conv2d(num_bands, c1, 1, bias=False),
            _bn(c1),
            nn.GELU(),
        )
        self.stage1 = SelectiveLargeKernelBlock(c1, dropout=dropout)
        self.mixstyle = GroupMixStyle(groups=4, p=mixstyle_p)
        self.down1 = ConvBNAct(c1, c2, 3, stride=2)
        self.stage2 = SelectiveLargeKernelBlock(c2, dropout=dropout)
        self.down2 = ConvBNAct(c2, c3, 3, stride=2)
        self.stage3 = SelectiveLargeKernelBlock(c3, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.spectral_projection(x)
        x = self.stage1(x)
        x = self.mixstyle(x)
        x = self.down1(x)
        x = self.stage2(x)
        x = self.down2(x)
        return self.stage3(x)


class PCAHSIEncoder(nn.Module):
    """PCA20 semantic front-end followed by the original HSI backbone.

    The PCA transform is frozen and stored in the checkpoint.  The degradation
    branch may still inspect the pre-PCA HSI so that PCA does not erase artifact
    evidence used for routing.
    """

    def __init__(self, raw_bands: int, pca_bands: int, dim: int = 96, dropout: float = 0.05, mixstyle_p: float = 0.0) -> None:
        super().__init__()
        self.pca = FixedPCATransform(raw_bands, pca_bands)
        self.backbone = HSIBackbone(pca_bands, dim, dropout, mixstyle_p)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(self.pca(x))


class LiDARBackbone(nn.Module):
    def __init__(self, in_channels: int, dim: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        c1, c2, c3 = 32, 64, int(dim)
        self.stem = ConvBNAct(in_channels, c1, 3, stride=1)
        self.stage1 = SelectiveLargeKernelBlock(c1, dropout=dropout)
        self.down1 = ConvBNAct(c1, c2, 3, stride=2)
        self.stage2 = SelectiveLargeKernelBlock(c2, dropout=dropout)
        self.down2 = ConvBNAct(c2, c3, 3, stride=2)
        self.stage3 = SelectiveLargeKernelBlock(c3, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stage1(self.stem(x))
        x = self.stage2(self.down1(x))
        return self.stage3(self.down2(x))


class TokenFFN(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(self.norm(x))


class BidirectionalCrossAttention(nn.Module):
    """Symmetric bidirectional cross-modal interaction used by Clean Fusion backbone."""

    def __init__(self, dim: int = 96, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.h_pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.l_pos = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.h_norm = nn.LayerNorm(dim)
        self.l_norm = nn.LayerNorm(dim)
        self.h_from_l = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.l_from_h = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.h_scale = nn.Parameter(torch.tensor(0.5))
        self.l_scale = nn.Parameter(torch.tensor(0.5))
        self.h_ffn = TokenFFN(dim, 2.0, dropout)
        self.l_ffn = TokenFFN(dim, 2.0, dropout)

    @staticmethod
    def _tokens(x: Tensor) -> Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _map(x: Tensor, h: int, w: int) -> Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)

    def forward(self, h_map: Tensor, l_map: Tensor) -> Tuple[Tensor, Tensor]:
        if h_map.shape[-2:] != l_map.shape[-2:]:
            raise ValueError(f"Cross-attention spatial mismatch: {h_map.shape} vs {l_map.shape}")
        height, width = h_map.shape[-2:]
        h = self._tokens(h_map + self.h_pos(h_map))
        l = self._tokens(l_map + self.l_pos(l_map))
        hn, ln = self.h_norm(h), self.l_norm(l)
        mh, _ = self.h_from_l(hn, ln, ln, need_weights=False)
        ml, _ = self.l_from_h(ln, hn, hn, need_weights=False)
        h = self.h_ffn(h + torch.tanh(self.h_scale) * mh)
        l = self.l_ffn(l + torch.tanh(self.l_scale) * ml)
        return self._map(h, height, width), self._map(l, height, width)


class CrossModalFusion(nn.Module):
    def __init__(self, dim: int = 96, mode: str = "cross_attention", heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.mode = str(mode)
        if self.mode not in {"cross_attention", "concat"}:
            raise ValueError("fusion mode must be cross_attention or concat")
        self.cross = BidirectionalCrossAttention(dim, heads, dropout) if self.mode == "cross_attention" else None
        self.mix = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            _bn(dim),
            nn.GELU(),
            SelectiveLargeKernelBlock(dim, expansion=2.0, dropout=dropout * 0.5),
        )

    def forward(self, h: Tensor, l: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        if self.cross is not None:
            h, l = self.cross(h, l)
        fused = self.mix(torch.cat([h, l, torch.abs(h - l), h * l], dim=1))
        return h, l, fused


class CenterGlobalPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        avg = x.mean(dim=(-2, -1))
        center = x[..., x.shape[-2] // 2, x.shape[-1] // 2]
        return self.proj(torch.cat([center, avg], dim=-1))


class ClassifierHead(nn.Module):
    def __init__(self, dim: int, classes: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(self.drop(self.norm(x)))


class CleanFusionBackbone(nn.Module):
    """Cross-Modal Representation Backbone (Clean Fusion backbone) for HSI-LiDAR classification.

    Architecture is unchanged from the validated Clean Fusion backbone implementation.
    """

    def __init__(
        self,
        num_bands: int,
        lidar_channels: int,
        num_classes: int,
        *,
        dim: int = 96,
        heads: int = 4,
        dropout: float = 0.1,
        fusion: str = "cross_attention",
        mixstyle_p: float = 0.25,
        pca_components: int = 0,
    ) -> None:
        super().__init__()
        self.pca_components = int(pca_components)
        self.hsi_encoder = (
            PCAHSIEncoder(num_bands, self.pca_components, dim, dropout * 0.5, mixstyle_p)
            if self.pca_components > 0 else HSIBackbone(num_bands, dim, dropout * 0.5, mixstyle_p)
        )
        self.lidar_encoder = LiDARBackbone(lidar_channels, dim, dropout * 0.5)
        self.fusion = CrossModalFusion(dim, fusion, heads, dropout)
        self.pool_h = CenterGlobalPool(dim, dropout)
        self.pool_l = CenterGlobalPool(dim, dropout)
        self.pool_f = CenterGlobalPool(dim, dropout)
        self.hsi_classifier = ClassifierHead(dim, num_classes, dropout)
        self.lidar_classifier = ClassifierHead(dim, num_classes, dropout)
        self.classifier = ClassifierHead(dim, num_classes, dropout)
        self.proj_h = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.proj_l = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))

    @torch.no_grad()
    def set_hsi_pca(self, mean, components) -> None:
        if not isinstance(self.hsi_encoder, PCAHSIEncoder):
            raise RuntimeError("CleanFusionBackbone was not constructed with PCA enabled")
        self.hsi_encoder.pca.set_parameters(mean, components)

    def forward(self, hsi: Tensor, lidar: Tensor) -> Dict[str, Tensor]:
        h_map = self.hsi_encoder(hsi)
        l_map = self.lidar_encoder(lidar)
        h_map, l_map, fused_map = self.fusion(h_map, l_map)
        h = self.pool_h(h_map)
        l = self.pool_l(l_map)
        fused = self.pool_f(fused_map)
        return {
            "hsi_map": h_map,
            "lidar_map": l_map,
            "fused_map": fused_map,
            "hsi_semantic": h,
            "lidar_semantic": l,
            "semantic": fused,
            "hsi_projection": F.normalize(self.proj_h(h), dim=-1),
            "lidar_projection": F.normalize(self.proj_l(l), dim=-1),
            "hsi_logits": self.hsi_classifier(h),
            "lidar_logits": self.lidar_classifier(l),
            "logits": self.classifier(fused),
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

