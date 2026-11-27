from __future__ import annotations

import copy
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.clean_fusion import CleanFusionBackbone


def _group_count(channels: int) -> int:
    groups = min(8, int(channels))
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return groups


class ModalityDegradationStatistics(nn.Module):
    """Modality-local statistics used by prototype-guided degradation representation."""

    output_dim = 16

    @staticmethod
    def gradients(x: Tensor) -> Tuple[Tensor, Tensor]:
        gx = torch.zeros_like(x)
        gy = torch.zeros_like(x)
        gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
        gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    def forward(self, x: Tensor) -> Tensor:
        gx, gy = self.gradients(x)
        grad = (gx.abs() + gy.abs()).mean(dim=1, keepdim=True)
        row_mean = x.mean(dim=(1, 3))
        col_mean = x.mean(dim=(1, 2))
        channel_mean = x.mean(dim=(2, 3))
        if x.shape[1] > 1:
            adjacent = (channel_mean[:, 1:] - channel_mean[:, :-1]).abs().mean(dim=1)
            channel_var = channel_mean.std(dim=1, unbiased=False)
        else:
            adjacent = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            channel_var = torch.zeros_like(adjacent)
        center = x[..., x.shape[-2] // 2, x.shape[-1] // 2].mean(dim=1)
        stats = torch.stack(
            [
                x.mean(dim=(1, 2, 3)),
                x.std(dim=(1, 2, 3), unbiased=False),
                x.abs().mean(dim=(1, 2, 3)),
                x.amax(dim=(1, 2, 3)),
                x.amin(dim=(1, 2, 3)),
                grad.mean(dim=(1, 2, 3)),
                grad.std(dim=(1, 2, 3), unbiased=False),
                gx.abs().mean(dim=(1, 2, 3)),
                gy.abs().mean(dim=(1, 2, 3)),
                row_mean.std(dim=1, unbiased=False),
                col_mean.std(dim=1, unbiased=False),
                (x.abs() < 1e-6).float().mean(dim=(1, 2, 3)),
                adjacent,
                channel_var,
                center,
                (x - x.mean(dim=(-2, -1), keepdim=True)).abs().mean(dim=(1, 2, 3)),
            ],
            dim=1,
        )
        return torch.nan_to_num(stats, nan=0.0, posinf=10.0, neginf=-10.0)


class ModalityDegradationEncoder(nn.Module):
    """Known-degradation + clean encoder.

    The degradation encoder deliberately does NOT allocate a classifier output for the held-out
    degradation.  The held-out corruption must be represented using relationships to the known
    degradation prototypes, rather than benefiting from a randomly initialized
    output node for an unseen class.
    """

    def __init__(self, degradation_dim: int, num_known_types: int, dropout: float, num_tokens: int) -> None:
        super().__init__()
        self.num_known_types = int(num_known_types)
        self.clean_type_id = self.num_known_types
        self.statistics = ModalityDegradationStatistics()
        hidden = max(64, int(degradation_dim) * 2)
        self.statistics_encoder = nn.Sequential(
            nn.LayerNorm(self.statistics.output_dim),
            nn.Linear(self.statistics.output_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_embeddings = nn.Parameter(torch.empty(int(num_tokens), hidden))
        nn.init.normal_(self.token_embeddings, std=0.02)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, int(degradation_dim)),
        )
        self.type_head = nn.Linear(int(degradation_dim), self.num_known_types + 1)
        self.severity_head = nn.Sequential(nn.Linear(int(degradation_dim), 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        statistics = self.statistics(x)
        context = self.statistics_encoder(statistics)
        hidden = context[:, None, :] + self.token_embeddings[None, :, :]
        tokens = F.normalize(self.token_projection(hidden), dim=-1)
        code = F.normalize(tokens.mean(dim=1), dim=-1)
        type_logits = self.type_head(code)
        clean_probability = F.softmax(type_logits, dim=1)[:, self.clean_type_id]
        return {
            "statistics": statistics,
            "tokens": tokens,
            "code": code,
            "type_logits": type_logits,
            "severity": self.severity_head(code).squeeze(1),
            "clean_probability": clean_probability,
        }


class FactorizedSpectralSpatialBlock(nn.Module):
    """Lightweight residual 3D block separating spatial and spectral filtering.

    This block is used only by the HSI degradation encoder.  It keeps the
    representation compact while retaining the local orientation and
    band-to-band structure that the global statistics cannot preserve.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.spatial = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )
        self.spectral = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.spectral(self.spatial(x))


class SpectralSpatialArtifactEncoder(nn.Module):
    """Compact HSI degradation encoder over learned spectral-spatial artifacts.

    Five content-normalized volumes are constructed from the HSI cube:
    local normalization, high-frequency residual, spectral first difference,
    and horizontal/vertical spatial differences.  The branch is adaptive to
    the number of HSI bands and patch size.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(5, 16, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.GroupNorm(_group_count(16), 16),
            nn.GELU(),
        )
        self.block1 = FactorizedSpectralSpatialBlock(16)
        self.down = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=(2, 2, 2), padding=1, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.GELU(),
        )
        self.block2 = FactorizedSpectralSpatialBlock(32)
        self.projection = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, int(output_dim)),
            nn.GELU(),
        )

    @staticmethod
    def artifact_volume(x: Tensor) -> Tensor:
        eps = 1e-5
        band_mean = x.mean(dim=(-2, -1), keepdim=True)
        band_std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
        local = (x - band_mean) / band_std
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high = (x - low) / band_std
        spectral = torch.zeros_like(x)
        if x.shape[1] > 1:
            spectral[:, 1:] = x[:, 1:] - x[:, :-1]
            spectral = spectral / band_std
        gx, gy = ModalityDegradationStatistics.gradients(x)
        gx = gx / band_std
        gy = gy / band_std
        volume = torch.stack([local, high, spectral, gx, gy], dim=1)
        return torch.nan_to_num(volume, nan=0.0, posinf=8.0, neginf=-8.0).clamp_(-8.0, 8.0)

    def forward(self, x: Tensor) -> Tensor:
        y = self.artifact_volume(x)
        y = self.block1(self.stem(y))
        y = self.block2(self.down(y))
        avg = F.adaptive_avg_pool3d(y, 1).flatten(1)
        mx = F.adaptive_max_pool3d(y, 1).flatten(1)
        return self.projection(torch.cat([avg, mx], dim=1))


class SpatialFrequencyArtifactEncoder(nn.Module):
    """Lightweight frequency branch for HSI degradation representation.

    The FFT is applied to a local high-frequency residual rather than the raw
    patch to reduce semantic-content leakage.  Mean and standard deviation of
    log magnitude across bands provide a fixed two-channel frequency map.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(16), 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(32), 32),
            nn.GELU(),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, int(output_dim)),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # torch.fft on CUDA is most portable in float32; cast back before the CNN.
        xf = x.float()
        low = F.avg_pool2d(xf, kernel_size=3, stride=1, padding=1)
        residual = xf - low
        residual = residual - residual.mean(dim=(-2, -1), keepdim=True)
        spectrum = torch.fft.rfft2(residual, dim=(-2, -1), norm="ortho")
        magnitude = torch.log1p(spectrum.abs())
        freq_mean = magnitude.mean(dim=1)
        freq_std = magnitude.std(dim=1, unbiased=False)
        freq = torch.stack([freq_mean, freq_std], dim=1)
        freq = torch.nan_to_num(freq, nan=0.0, posinf=10.0, neginf=0.0).to(dtype=x.dtype)
        y = self.encoder(freq)
        avg = F.adaptive_avg_pool2d(y, 1).flatten(1)
        mx = F.adaptive_max_pool2d(y, 1).flatten(1)
        return self.projection(torch.cat([avg, mx], dim=1))


class HSIMultiBranchDegradationEncoder(nn.Module):
    """HSI degradation encoder with configurable evidence branches.

    Variants share the flat type/severity heads and the same
    degradation-code dimensionality.  This keeps prototype fitting, experts and
    repair-utility/Identity routing directly comparable across the .1 ablation.

    Supported variants:
      - stats_only:     statistics branch only
      - stats_3d:      statistics + spectral-spatial 3D branch
      - stats_fft:     statistics + frequency branch
      - stats_3d_fft:  statistics + 3D + frequency ()
    """

    VALID_VARIANTS = {"stats_only", "stats_3d", "stats_fft", "stats_3d_fft"}

    def __init__(
        self,
        degradation_dim: int,
        num_known_types: int,
        dropout: float,
        num_tokens: int,
        *,
        variant: str = "stats_3d_fft",
    ) -> None:
        super().__init__()
        variant = str(variant).strip().lower()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown HSI degradation encoder variant: {variant}")
        self.variant = variant
        self.num_known_types = int(num_known_types)
        self.clean_type_id = self.num_known_types
        self.statistics = ModalityDegradationStatistics()
        hidden = max(64, int(degradation_dim) * 2)
        self.statistics_encoder = nn.Sequential(
            nn.LayerNorm(self.statistics.output_dim),
            nn.Linear(self.statistics.output_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.use_3d = variant in {"stats_3d", "stats_3d_fft"}
        self.use_fft = variant in {"stats_fft", "stats_3d_fft"}
        self.spectral_spatial_encoder = SpectralSpatialArtifactEncoder(hidden) if self.use_3d else None
        self.frequency_encoder = SpatialFrequencyArtifactEncoder(hidden) if self.use_fft else None
        branch_count = 1 + int(self.use_3d) + int(self.use_fft)
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden * branch_count),
            nn.Linear(hidden * branch_count, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ) if branch_count > 1 else nn.Identity()
        self.token_embeddings = nn.Parameter(torch.empty(int(num_tokens), hidden))
        nn.init.normal_(self.token_embeddings, std=0.02)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, int(degradation_dim)),
        )
        self.type_head = nn.Linear(int(degradation_dim), self.num_known_types + 1)
        self.severity_head = nn.Sequential(nn.Linear(int(degradation_dim), 1), nn.Sigmoid())

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        statistics = self.statistics(x)
        features = [self.statistics_encoder(statistics)]
        out: Dict[str, Tensor] = {"statistics": statistics}
        if self.use_3d:
            spectral_spatial_feature = self.spectral_spatial_encoder(x)
            features.append(spectral_spatial_feature)
            out["spectral_spatial_feature"] = spectral_spatial_feature
        if self.use_fft:
            frequency_feature = self.frequency_encoder(x)
            features.append(frequency_feature)
            out["frequency_feature"] = frequency_feature
        context = features[0] if len(features) == 1 else self.fusion(torch.cat(features, dim=1))
        hidden = context[:, None, :] + self.token_embeddings[None, :, :]
        tokens = F.normalize(self.token_projection(hidden), dim=-1)
        code = F.normalize(tokens.mean(dim=1), dim=-1)
        type_logits = self.type_head(code)
        clean_probability = F.softmax(type_logits, dim=1)[:, self.clean_type_id]
        out.update({
            "tokens": tokens,
            "code": code,
            "type_logits": type_logits,
            "severity": self.severity_head(code).squeeze(1),
            "clean_probability": clean_probability,
        })
        return out


# Alias retained for the spectral-spatial-frequency encoder variant.
class HSISpectralSpatialFrequencyDegradationEncoder(HSIMultiBranchDegradationEncoder):
    def __init__(self, degradation_dim: int, num_known_types: int, dropout: float, num_tokens: int) -> None:
        super().__init__(degradation_dim, num_known_types, dropout, num_tokens, variant="stats_3d_fft")


class ResidualRepairExpert(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float, residual_scale: float) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.trunk = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden_dim), hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Conv2d(hidden_dim, feature_dim, 3, padding=1)
        self.mask = nn.Conv2d(hidden_dim, 1, 1)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, base: Tensor) -> Dict[str, Tensor]:
        hidden = self.trunk(base)
        delta = torch.tanh(self.delta(hidden)) * self.residual_scale
        mask = torch.sigmoid(self.mask(hidden))
        return {"feature": base + mask * delta, "delta": delta, "mask": mask}


class DegradationPrototypeBank(nn.Module):
    """Deterministic spherical multi-support prototypes for known degradations."""

    def __init__(self, num_known_types: int, severity_slots: int, code_dim: int, num_subprototypes: int, kmeans_iterations: int) -> None:
        super().__init__()
        self.num_known_types = int(num_known_types)
        self.clean_type_id = self.num_known_types
        self.num_types = self.num_known_types + 1
        self.severity_slots = int(severity_slots)
        self.code_dim = int(code_dim)
        self.num_subprototypes = int(num_subprototypes)
        self.kmeans_iterations = int(kmeans_iterations)
        shape = (self.num_types, self.severity_slots, self.num_subprototypes, self.code_dim)
        self.register_buffer("prototypes", torch.zeros(shape))
        self.register_buffer("initialized", torch.zeros(shape[:-1], dtype=torch.bool))

    def severity_to_slot(self, type_id: int, severity: int) -> int:
        if int(type_id) == self.clean_type_id:
            return 0
        return max(0, min(self.severity_slots - 1, int(severity) - 1))

    @torch.no_grad()
    def reset(self) -> None:
        self.prototypes.zero_(); self.initialized.zero_()

    @torch.no_grad()
    def _centers(self, points: Tensor) -> Tensor:
        points = F.normalize(points, dim=-1)
        if points.shape[0] == 0:
            raise ValueError("Cannot fit prototypes from zero points")
        mean = F.normalize(points.mean(dim=0), dim=0)
        selected = [int(torch.argmax(points @ mean))]
        while len(selected) < min(self.num_subprototypes, points.shape[0]):
            chosen = points[selected]
            d = (1.0 - points @ chosen.t()).min(dim=1).values
            d[selected] = -1.0
            selected.append(int(torch.argmax(d)))
        centers = points[selected]
        if centers.shape[0] < self.num_subprototypes:
            repeats = (self.num_subprototypes + centers.shape[0] - 1) // centers.shape[0]
            centers = centers.repeat(repeats, 1)[:self.num_subprototypes]
        centers = F.normalize(centers, dim=-1)
        old_assign = None
        for _ in range(max(1, self.kmeans_iterations)):
            assign = torch.argmax(points @ centers.t(), dim=1)
            new = []
            for k in range(self.num_subprototypes):
                pts = points[assign == k]
                new.append(centers[k] if pts.numel() == 0 else F.normalize(pts.mean(dim=0), dim=0))
            centers = F.normalize(torch.stack(new), dim=-1)
            if old_assign is not None and torch.equal(assign, old_assign):
                break
            old_assign = assign
        return centers

    @torch.no_grad()
    def fit_condition(self, tokens: Tensor, type_id: int, severity: int) -> None:
        if tokens.ndim == 3:
            tokens = tokens.reshape(-1, tokens.shape[-1])
        slot = self.severity_to_slot(type_id, severity)
        self.prototypes[int(type_id), slot].copy_(self._centers(tokens.detach()))
        self.initialized[int(type_id), slot].fill_(True)

    def degradation_conditions(self) -> Tuple[Tensor, Tensor]:
        valid = self.initialized[:self.num_known_types].all(dim=-1)
        if not bool(valid.any()):
            raise RuntimeError("No degradation prototypes are initialized")
        types = torch.arange(self.num_known_types, device=self.prototypes.device)[:, None].expand_as(valid)
        protos = self.prototypes[:self.num_known_types][valid]
        return protos, types[valid]

    def family_centroids(self) -> Tensor:
        """Return one normalized centroid per known expert family.

        Only initialized severity/sub-prototypes contribute.  This keeps the
        repairability scorer independent of the number of severity slots and
        therefore identical across Houston, Trento and MUUFL.
        """
        rows = []
        for type_id in range(self.num_known_types):
            valid = self.initialized[type_id]
            points = self.prototypes[type_id][valid]
            if points.numel() == 0:
                raise RuntimeError(f"No prototype is initialized for expert family {type_id}")
            rows.append(F.normalize(points.mean(dim=0), dim=0))
        return torch.stack(rows, dim=0)


class ExpertConditionedRepairUtility(nn.Module):
    """Shared scorer that predicts whether each residual expert is useful.

    The same MLP is applied to every expert.  Consequently the module supports
    a different number of known HSI/LiDAR degradations without dataset-specific
    output heads.  Its target is a measured *classification/feature gain over
    Identity*, not a degradation-type label.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        degradation_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # code, prototype, |difference|, elementwise agreement, pooled base,
        # pooled proposed expert residual,
        # cosine similarity, prototype routing weight, routing entropy,
        # clean probability, estimated severity and residual magnitude.
        input_dim = 4 * int(degradation_dim) + 2 * int(feature_dim) + 6
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )
        # Start exactly at the Identity boundary so the prototype-cosine path
        # is unchanged until the utility head has completed its own training.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        *,
        code: Tensor,
        base: Tensor,
        expert_features: Tensor,
        centroids: Tensor,
        similarity: Tensor,
        family_weights: Tensor,
        route_entropy: Tensor,
        clean_probability: Tensor,
        severity: Tensor,
    ) -> Tensor:
        batch, experts = family_weights.shape
        code_e = code[:, None, :].expand(batch, experts, -1)
        proto_e = centroids[None, :, :].expand(batch, experts, -1)
        pooled = F.adaptive_avg_pool2d(base, 1).flatten(1)
        pooled = F.layer_norm(pooled, (pooled.shape[1],))
        pooled_e = pooled[:, None, :].expand(batch, experts, -1)
        residual = expert_features - base[:, None]
        residual_flat = residual.reshape(batch * experts, *residual.shape[2:])
        residual_pooled = F.adaptive_avg_pool2d(residual_flat, 1).flatten(1)
        residual_pooled = F.layer_norm(residual_pooled, (residual_pooled.shape[1],))
        residual_pooled = residual_pooled.reshape(batch, experts, -1)
        scalars = torch.stack(
            [
                similarity,
                family_weights,
                route_entropy[:, None].expand(batch, experts),
                clean_probability[:, None].expand(batch, experts),
                severity[:, None].expand(batch, experts),
                residual.abs().mean(dim=(2, 3, 4)),
            ],
            dim=-1,
        )
        inputs = torch.cat(
            [code_e, proto_e, (code_e - proto_e).abs(), code_e * proto_e, pooled_e, residual_pooled, scalars],
            dim=-1,
        )
        return self.network(inputs).squeeze(-1)


class PrototypeGuidedRepairBranch(nn.Module):
    """Prototype-guided expert composition with a clean-preserving gate.

    prototype-guided routing uses the explicitly supervised clean probability as a base-feature
    gate, while cosine prototypes decide how non-clean repair mass is
    distributed across known experts.
    """

    def __init__(
        self,
        num_known_types: int,
        *,
        feature_dim: int,
        degradation_dim: int,
        dropout: float,
        num_tokens: int,
        num_subprototypes: int,
        severity_slots: int,
        kmeans_iterations: int,
        cosine_temperature: float,
        expert_hidden_dim: int,
        expert_residual_scale: float,
        repair_hidden_dim: int,
        repair_dropout: float,
        repair_gain_weight: float,
        repair_activation_threshold: float,
        repair_gate_temperature: float,
        degradation_encoder_variant: str = "statistics",
    ) -> None:
        super().__init__()
        self.num_known_types = int(num_known_types)
        self.clean_type_id = self.num_known_types
        self.temperature = float(cosine_temperature)
        self.repair_gain_weight = float(repair_gain_weight)
        self.repair_activation_threshold = float(repair_activation_threshold)
        self.repair_gate_temperature = float(repair_gate_temperature)
        raw_variant = str(degradation_encoder_variant).strip().lower()
        aliases = {
            "statistics": "stats_only",
            "spectral_spatial_frequency": "stats_3d_fft",
            "hsi_spectral_spatial_frequency": "stats_3d_fft",
        }
        variant = aliases.get(raw_variant, raw_variant)
        if variant == "stats_only":
            # Preserve the exact  statistics-only encoder for the baseline arm.
            self.degradation_encoder = ModalityDegradationEncoder(degradation_dim, num_known_types, dropout, num_tokens)
        elif variant in HSIMultiBranchDegradationEncoder.VALID_VARIANTS:
            self.degradation_encoder = HSIMultiBranchDegradationEncoder(
                degradation_dim, num_known_types, dropout, num_tokens, variant=variant
            )
        else:
            raise ValueError(f"Unknown degradation encoder variant: {degradation_encoder_variant}")
        self.degradation_encoder_variant = variant
        self.prototype_bank = DegradationPrototypeBank(num_known_types, severity_slots, degradation_dim, num_subprototypes, kmeans_iterations)
        self.experts = nn.ModuleList([
            ResidualRepairExpert(feature_dim, expert_hidden_dim, dropout, expert_residual_scale)
            for _ in range(self.num_known_types)
        ])

        self.repair_utility = ExpertConditionedRepairUtility(
            feature_dim=feature_dim,
            degradation_dim=degradation_dim,
            hidden_dim=repair_hidden_dim,
            dropout=repair_dropout,
        )

    def _family_weights(self, code: Tensor) -> Tuple[Tensor, Tensor]:
        protos, types = self.prototype_bank.degradation_conditions()
        centroids = F.normalize(protos.mean(dim=1), dim=-1)
        similarity = F.normalize(code, dim=-1) @ centroids.t()
        condition = F.softmax(similarity / max(self.temperature, 1e-6), dim=1)
        family = []
        for t in range(self.num_known_types):
            mask = types == t
            family.append(condition[:, mask].sum(dim=1))
        family = torch.stack(family, dim=1)
        family = family / family.sum(dim=1, keepdim=True).clamp_min(1e-12)
        entropy = -(condition.clamp_min(1e-12) * condition.clamp_min(1e-12).log()).sum(dim=1)
        return family, entropy

    def describe(self, x: Tensor, base: Tensor, *, include_experts: bool = True) -> Dict[str, Tensor]:
        """Build the fixed degradation/expert state used by routing and utility training."""
        deg = self.degradation_encoder(x)
        family, entropy = self._family_weights(deg["code"])
        centroids = self.prototype_bank.family_centroids()
        similarity = F.normalize(deg["code"], dim=-1) @ centroids.t()
        state = {
            **deg,
            "family_weights": family,
            "route_entropy": entropy,
            "family_centroids": centroids,
            "family_similarity": similarity,
        }
        if include_experts:
            state["expert_features"] = torch.stack(
                [expert(base)["feature"] for expert in self.experts], dim=1
            )
        return state

    def predict_repair_utility(self, state: Dict[str, Tensor], base: Tensor) -> Tensor:
        return self.repair_utility(
            code=state["code"],
            base=base,
            expert_features=state["expert_features"],
            centroids=state["family_centroids"],
            similarity=state["family_similarity"],
            family_weights=state["family_weights"],
            route_entropy=state["route_entropy"],
            clean_probability=state["clean_probability"],
            severity=state["severity"],
        )

    @staticmethod
    def clean_aware_feature(state: Dict[str, Tensor], base: Tensor) -> Tensor:
        family = state["family_weights"]
        expert_mix = (family[:, :, None, None, None] * state["expert_features"]).sum(dim=1)
        clean_prob = state["clean_probability"].clamp(0.0, 1.0)
        return clean_prob[:, None, None, None] * base + (1.0 - clean_prob)[:, None, None, None] * expert_mix

    def route_and_restore(self, x: Tensor, base: Tensor, route: str = "cosine_experts") -> Dict[str, Tensor]:
        if route not in {"base_only", "cosine_experts", "repairability"}:
            raise ValueError(f"Unknown repair route: {route}")
        deg = self.degradation_encoder(x)
        utility = base.new_zeros((base.shape[0], self.num_known_types))
        selected_utility = base.new_zeros((base.shape[0],))
        identity_mass = base.new_ones((base.shape[0],))
        if route == "base_only":
            family = base.new_zeros((base.shape[0], self.num_known_types))
            entropy = base.new_zeros((base.shape[0],))
            feature = base
            repair_mass = base.new_zeros((base.shape[0],))
        else:
            family, entropy = self._family_weights(deg["code"])
            experts = torch.stack([expert(base)["feature"] for expert in self.experts], dim=1)
            clean_prob = deg["clean_probability"].clamp(0.0, 1.0)
            if route == "repairability":
                centroids = self.prototype_bank.family_centroids()
                similarity = F.normalize(deg["code"], dim=-1) @ centroids.t()
                state = {
                    **deg,
                    "family_weights": family,
                    "route_entropy": entropy,
                    "family_centroids": centroids,
                    "family_similarity": similarity,
                    "expert_features": experts,
                }
                utility = self.predict_repair_utility(state, base)
                # The Identity option has fixed utility 0.  A smooth gate avoids
                # brittle all-or-nothing decisions close to that boundary.
                selected_utility = utility.max(dim=1).values
                useful = torch.sigmoid(
                    (selected_utility - self.repair_activation_threshold)
                    / max(self.repair_gate_temperature, 1e-6)
                )
                adjusted = family.clamp_min(1e-12).log() + self.repair_gain_weight * utility
                family = F.softmax(adjusted, dim=1)
                repair_mass = (1.0 - clean_prob) * useful
            else:
                repair_mass = 1.0 - clean_prob
            expert_mix = (family[:, :, None, None, None] * experts).sum(dim=1)
            if route == "repairability":
                # Rejected repair mass must return to Identity, not shrink the
                # representation.  Expressing the result as base + gated delta
                # makes that invariant explicit.
                feature = base + repair_mass[:, None, None, None] * (expert_mix - base)
            else:
                feature = clean_prob[:, None, None, None] * base + repair_mass[:, None, None, None] * expert_mix
            identity_mass = 1.0 - repair_mass
        return {
            "feature": feature,
            "weights": family,
            "clean_probability": deg["clean_probability"],
            "repair_mass": repair_mass,
            "identity_mass": identity_mass,
            "repair_utility": utility,
            "expert_features": experts if route != "base_only" else None,
            "selected_repair_utility": selected_utility,
            "route_entropy": entropy,
            "type_logits": deg["type_logits"],
            "severity": deg["severity"],
            "tokens": deg["tokens"],
            "code": deg["code"],
        }


class PCRFusion(nn.Module):
    """PCRFusion: prototype-guided routing and compositional residual repair."""

    def __init__(
        self,
        clean_fusion: CleanFusionBackbone,
        hsi_known_count: int,
        lidar_known_count: int,
        *,
        feature_dim: int,
        degradation_dim: int = 32,
        degradation_dropout: float = 0.10,
        num_tokens: int = 8,
        num_subprototypes: int = 4,
        severity_slots: int = 3,
        kmeans_iterations: int = 20,
        cosine_temperature: float = 0.10,
        expert_hidden_dim: int = 48,
        expert_residual_scale: float = 0.50,
        repair_hidden_dim: int = 64,
        repair_dropout: float = 0.10,
        repair_gain_weight: float = 2.0,
        repair_activation_threshold: float = 0.02,
        repair_gate_temperature: float = 0.03,
        hsi_degradation_encoder_variant: str = "stats_only",
        lidar_degradation_encoder_variant: str = "statistics",
        shared_expert_per_modality: bool = False,
        composition_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.hsi_encoder = copy.deepcopy(clean_fusion.hsi_encoder)
        self.lidar_encoder = copy.deepcopy(clean_fusion.lidar_encoder)
        self.fusion = copy.deepcopy(clean_fusion.fusion)
        self.pool_f = copy.deepcopy(clean_fusion.pool_f)
        self.classifier = copy.deepcopy(clean_fusion.classifier)
        self.register_buffer("repairability_ready", torch.tensor(False, dtype=torch.bool))
        self.hsi_repair_branch = PrototypeGuidedRepairBranch(
            hsi_known_count, feature_dim=feature_dim, degradation_dim=degradation_dim,
            dropout=degradation_dropout, num_tokens=num_tokens, num_subprototypes=num_subprototypes,
            severity_slots=severity_slots, kmeans_iterations=kmeans_iterations,
            cosine_temperature=cosine_temperature, expert_hidden_dim=expert_hidden_dim,
            expert_residual_scale=expert_residual_scale,
            repair_hidden_dim=repair_hidden_dim, repair_dropout=repair_dropout,
            repair_gain_weight=repair_gain_weight,
            repair_activation_threshold=repair_activation_threshold,
            repair_gate_temperature=repair_gate_temperature,
            degradation_encoder_variant=hsi_degradation_encoder_variant,
        )
        self.lidar_repair_branch = PrototypeGuidedRepairBranch(
            lidar_known_count, feature_dim=feature_dim, degradation_dim=degradation_dim,
            dropout=degradation_dropout, num_tokens=num_tokens, num_subprototypes=num_subprototypes,
            severity_slots=severity_slots, kmeans_iterations=kmeans_iterations,
            cosine_temperature=cosine_temperature, expert_hidden_dim=expert_hidden_dim,
            expert_residual_scale=expert_residual_scale,
            repair_hidden_dim=repair_hidden_dim, repair_dropout=repair_dropout,
            repair_gain_weight=repair_gain_weight,
            repair_activation_threshold=repair_activation_threshold,
            repair_gate_temperature=repair_gate_temperature,
            degradation_encoder_variant=lidar_degradation_encoder_variant,
        )
        if shared_expert_per_modality:
            raise ValueError('This release requires independent family experts for each modality.')

        from repair.low_mix import install
        install(self, composition_config, int(feature_dim))

    def set_repairability_ready(self, ready: bool = True) -> None:
        self.repairability_ready.fill_(bool(ready))

    def active_route(self, route: str) -> str:
        if route == "auto":
            return "top1"
        return route

    def classify_features(self, hsi_feature: Tensor, lidar_feature: Tensor) -> Dict[str, Tensor]:
        h_cross, l_cross, fused_map = self.fusion(hsi_feature, lidar_feature)
        semantic = self.pool_f(fused_map)
        return {
            "logits": self.classifier(semantic),
            "semantic": semantic,
            "fused_map": fused_map,
            "hsi_cross": h_cross,
            "lidar_cross": l_cross,
        }

    def forward(self, hsi: Tensor, lidar: Tensor, route: str = "auto") -> Dict[str, Tensor]:
        route = self.active_route(route)
        h_base = self.hsi_encoder(hsi)
        l_base = self.lidar_encoder(lidar)
        h = self.hsi_repair_branch.route_and_restore(hsi, h_base, route=route)
        l = self.lidar_repair_branch.route_and_restore(lidar, l_base, route=route)
        fused = self.classify_features(h["feature"], l["feature"])
        return {
            "logits": fused["logits"],
            "semantic": fused["semantic"],
            "fused_map": fused["fused_map"],
            "hsi_base": h_base,
            "lidar_base": l_base,
            "hsi_feature": h["feature"],
            "lidar_feature": l["feature"],
            "hsi_clean_probability": h["clean_probability"],
            "lidar_clean_probability": l["clean_probability"],
            "hsi_repair_mass": h["repair_mass"],
            "lidar_repair_mass": l["repair_mass"],
            "hsi_identity_mass": h["identity_mass"],
            "lidar_identity_mass": l["identity_mass"],
            "hsi_repair_utility": h["repair_utility"],
            "lidar_repair_utility": l["repair_utility"],
            "hsi_selected_repair_utility": h["selected_repair_utility"],
            "lidar_selected_repair_utility": l["selected_repair_utility"],
            "hsi_weights": h["weights"],
            "lidar_weights": l["weights"],
            "hsi_type_logits": h["type_logits"],
            "lidar_type_logits": l["type_logits"],
            "hsi_severity": h["severity"],
            "lidar_severity": l["severity"],
        }


def load_clean_fusion_from_checkpoint(checkpoint: dict, num_bands: int, lidar_channels: int, num_classes: int) -> CleanFusionBackbone:
    args = dict(checkpoint.get("args", {}))
    model = CleanFusionBackbone(
        num_bands,
        lidar_channels,
        num_classes,
        dim=int(args.get("dim", 96)),
        heads=int(args.get("heads", 4)),
        dropout=float(args.get("dropout", 0.10)),
        fusion=str(args.get("fusion", "cross_attention")),
        mixstyle_p=float(args.get("mixstyle_p", 0.0)),
        pca_components=int(args.get("pca_components", 0)),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model


def count_parameters(model: nn.Module) -> int:
    """Total parameter count (independent of current freeze state)."""
    return sum(p.numel() for p in model.parameters())
