"""The tested Full low-mix training path, with Top1 as final inference."""
from __future__ import annotations
import copy
from types import MethodType
import torch
import torch.nn.functional as F

DEFAULTS = dict(enabled=True, mode='prototype', hidden_dim=64, channel_groups=8,
                spatial_grid=3, train_topk=0, mix_weight=0.05, route_weight=0.1,
                feature_weight=1.0, semantic_weight=0.35, classification_weight=0.25,
                teacher_target='none', teacher_temperature=0.5, oracle_steps=30,
                warmup_fraction=0.2, ramp_fraction=0.2, detach_clean_gate=True,
                router_learning_rate=8e-4)


def settings(raw=None):
    cfg = {**DEFAULTS, **copy.deepcopy(raw or {})}
    required = dict(enabled=True, mode='prototype', teacher_target='none', train_topk=0,
                    mix_weight=0.05, feature_weight=1.0, semantic_weight=0.35,
                    classification_weight=0.25, warmup_fraction=0.2,
                    ramp_fraction=0.2, detach_clean_gate=True)
    for key, expected in required.items():
        if cfg[key] != expected:
            raise ValueError(f'This single-model release requires {key}={expected!r}, got {cfg[key]!r}')
    return cfg


def install(model, raw, features):
    cfg = settings(raw)
    model.composition_config = cfg
    for branch in (model.hsi_repair_branch, model.lidar_repair_branch):
        branch.composition_config = cfg
        def routed(self, x, base, route='top1'):
            if route not in ('cosine_experts', 'top1'):
                raise ValueError(f'Unsupported route: {route}')
            deg = self.degradation_encoder(x)
            candidates = torch.stack([expert(base)['feature'] for expert in self.experts], 1)
            result = compose(self, deg, base, candidates, topk=1 if route == 'top1' else 0)
            zeros = base.new_zeros(base.shape[0])
            return {**deg, **result, 'expert_features': candidates,
                    'repair_utility': base.new_zeros(base.shape[0], self.num_known_types),
                    'selected_repair_utility': zeros}
        branch.route_and_restore = MethodType(routed, branch)


def distribution(branch, deg, base):
    prior, _ = branch._family_weights(deg['code'])
    return prior[:, :, None, None, None]


def retain(weights, k):
    k = min(int(k), weights.shape[1])
    if k <= 0 or k == weights.shape[1]: return weights
    ranking = weights.mean((2, 3, 4), keepdim=True)
    ids = ranking.topk(k, dim=1).indices
    mask = torch.zeros_like(ranking).scatter_(1, ids, 1).expand_as(weights)
    kept = weights * mask
    return kept / kept.sum(1, keepdim=True).clamp_min(1e-12)


def compose(branch, deg, base, candidates, *, topk=0, detach_gate=False):
    weights = retain(distribution(branch, deg, base), topk)
    mix = (weights * candidates).sum(1)
    mass = 1.0 - deg['clean_probability'].clamp(0, 1)
    if detach_gate: mass = mass.detach()
    feature = base + mass[:, None, None, None] * (mix - base)
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(1)
    return dict(feature=feature, weights=weights.expand_as(candidates).mean((2, 3, 4)),
                detailed_weights=weights, repair_mass=mass, identity_mass=1.0-mass,
                route_entropy=entropy.flatten(1).mean(1))


def ramp(epoch, epochs, cfg):
    warm = int(epochs * float(cfg['warmup_fraction']))
    if epoch <= warm: return 0.0
    duration = max(1, int(epochs * float(cfg['ramp_fraction'])))
    return min(1.0, (epoch-warm)/duration)


def additional_loss(model, epoch, epochs, h_deg, l_deg, h_base, l_base, h_candidates,
                    l_candidates, target_h, target_l, teacher_out, labels, h_clean, l_clean):
    cfg = model.composition_config
    factor = ramp(epoch, epochs, cfg)
    zero = h_base.new_zeros(())
    logs = {key:zero for key in ('composition_feature','composition_semantic','composition_classification','composition_route')}
    logs['composition_ramp'] = zero.new_tensor(factor)
    if factor == 0: return zero, logs
    hr = compose(model.hsi_repair_branch,h_deg,h_base,h_candidates,topk=0,detach_gate=cfg['detach_clean_gate'])
    lr = compose(model.lidar_repair_branch,l_deg,l_base,l_candidates,topk=0,detach_gate=cfg['detach_clean_gate'])
    logs['composition_feature'] = .5*(F.smooth_l1_loss(hr['feature'],target_h)+F.smooth_l1_loss(lr['feature'],target_l))
    fused = model.classify_features(hr['feature'],lr['feature'])
    logs['composition_semantic'] = F.smooth_l1_loss(fused['semantic'],teacher_out['semantic'].detach())
    logs['composition_classification'] = F.cross_entropy(fused['logits'],labels)
    mixed = (float(cfg['feature_weight'])*logs['composition_feature']+
             float(cfg['semantic_weight'])*logs['composition_semantic']+
             float(cfg['classification_weight'])*logs['composition_classification'])
    return factor*(float(cfg['mix_weight'])*mixed+float(cfg['route_weight'])*logs['composition_route']),logs
