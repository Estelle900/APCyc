#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Tuple
import inspect
import os
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
from torch.autograd import grad
from torch_scatter import scatter_mean

from utils.nn_utils import variadic_meshgrid
from models.LDM.diffusion.transition import construct_transition
from models.LDM.diffusion.dpm_full import EpsilonNet, low_trianguler_inv
from models.dyMEAN.modules.radial_basis import RadialBasis

from apcyc.pair_init import OnlinePairInit, get_anchor_coords
from apcyc.cyclization_heads import CyclizationHeads
from apcyc.cyclic_injection import EdgeFeatureInjector, BiasGateInjector, cyclic_lambda
from apcyc.guidance import (
    parse_guidance_w,
    clip_guidance_w,
    guidance_lambda,
    grad_norm_balance,
    sigma2_from_transition,
    energy_from_pred,
)
from apcyc.pocket_cross_attn import PocketCrossAttn

APCYC_VERBOSE = os.environ.get("APCYC_VERBOSE", "0") == "1"
max_h_step = float(os.environ.get("APCYC_MAX_H_UPDATE", "0"))

def trace_magnitude(name, x, t, mask=None):
    x = x.detach()

    if mask is not None:
        x = x[mask]

    if x.numel() == 0:
        return

    finite = torch.isfinite(x)
    peak = (
        x[finite].abs().max().item()
        if finite.any() else float("nan")
    )
    if not finite.all() or peak > 500:
        print(
            f"[APCyc magnitude] t={t} "
            f"{name}: max_abs={peak:.6g}, "
            f"nonfinite={(~finite).sum().item()}",
            flush=True,
        )

def check_particle_scale(
    name, x, t, batch_ids, mask_generate,
    limit=1e4,
):
    # Debugging limit in normalized latent units.
    # Not a physical coordinate-validity threshold.
    x_gen = x[mask_generate]
    ids = batch_ids[mask_generate]

    if x_gen.numel() == 0:
        return

    node_peaks = x_gen.detach().abs().flatten(1).amax(1)
    n_particles = int(batch_ids.max().item()) + 1

    peaks = torch.zeros(
        n_particles,
        device=x.device,
        dtype=node_peaks.dtype,
    )
    peaks.scatter_reduce_(
        0, ids, node_peaks,
        reduce="amax",
        include_self=True,
    )

    bad = (~torch.isfinite(peaks)) | (peaks > limit)

    if bad.any():
        particle_ids = bad.nonzero(as_tuple=True)[0].tolist()
        raise FloatingPointError(
            f"{name} at t={t}: "
            f"local_particles={particle_ids}, "
            f"max_abs={peaks[bad].tolist()}"
        )

import os
import torch


def clip_h_update(H_prev, H_next, mask_generate, max_norm):
    """
    Limit ||H_next[i] - H_prev[i]||_2 for each generated residue.

    H_prev, H_next: [N, hidden_dim]
    mask_generate:  [N]

    Does not modify receptor/context residues.
    """
    if max_norm <= 0:
        return H_next

    if not torch.isfinite(H_prev).all():
        raise FloatingPointError(
            "Non-finite H_prev before H update clipping"
        )

    if not torch.isfinite(H_next).all():
        raise FloatingPointError(
            "Non-finite H_next before H update clipping"
        )

    delta = H_next - H_prev
    norms = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    scale = (max_norm / norms.clamp(min=1e-12)).clamp(max=1.0)
    H_clipped = (H_prev.double() + delta * scale).to(H_next.dtype)

    clipped = (norms.squeeze(-1) > max_norm) & mask_generate.bool()

    if clipped.any():
        print(
            "[APCyc H CLIP]",
            f"max_update_norm="
            f"{norms[mask_generate].max().item():.4f}",
            f"clipped_residues={clipped.sum().item()}",
            flush=True,
        )

    return torch.where(
        mask_generate[:, None],
        H_clipped,
        H_next,
    )

class APCycFullDPM(nn.Module):
    def __init__(
        self,
        latent_size: int,
        hidden_size: int,
        n_channel: int,
        num_steps: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        trans_pos_type: str = 'Diffusion',
        trans_seq_type: str = 'Diffusion',
        trans_pos_opt: Dict = {},
        trans_seq_opt: Dict = {},
        n_rbf: int = 0,
        cutoff: float = 1.0,
        std: float = 10.0,
        additional_pos_embed: bool = True,
        dist_rbf: int = 0,
        dist_rbf_cutoff: float = 7.0,
        pair_dim: int = 128,
        pair_struct_dim: int = 64,
        pair_hidden_dim: int = 256,
        pair_rbf_dim: int = 32,
        pair_rbf_cutoff: float = 12.0,
        edge_type_dim: int = 16,
        pair_anchor_mode: str = 'ca',
        pair_anchor_channel_idx: int = 0,
        enable_covalent_edges: bool = False,
        covalent_edge_type_id: int = 3,
        covalent_edge_start_t: Optional[int] = None,
        covalent_edge_separate_params: bool = False,
        type_hidden: int = 128,
        pair_head_hidden: int = 128,
        min_sep_all: int = 1,
        min_sep_ss: int = 3,
        allow_ht_alt_pairs: bool = False,
        ht_alt_min_end: int = 1,
        mask_mode: str = 'teacher',
        relax_residue_constraints: bool = True,
        enable_pair_injection: bool = True,
        lambda_cyc_max: float = 1.0,
        lambda_cyc_schedule: str = 'linear',
        lambda_b_max: float = 1.0,
        lambda_g_max: float = 1.0,
        struct_proj_dim: int = 32,
        ca_idx: int = 1,
        pair_struct_refine: bool = False,
        pair_struct_refine_hidden: int = 64,
        pair_cyc_refine: bool = True,
        pair_cyc_refine_hidden: int = 64,
        enable_pocket_cross_attn: bool = False,
        pocket_cross_attn_dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.n_channel = n_channel
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.dropout = dropout
        pair_edge_dim = struct_proj_dim + 1 + 3 + 1 + edge_type_dim
        self.pair_edge_dim = pair_edge_dim
        self.eps_net = EpsilonNet(
            latent_size, hidden_size, n_channel, n_layers=n_layers, edge_size=dist_rbf + pair_edge_dim,
            n_rbf=n_rbf, cutoff=cutoff, dropout=dropout, additional_pos_embed=additional_pos_embed)
        if dist_rbf > 0:
            self.dist_rbf = RadialBasis(dist_rbf, dist_rbf_cutoff)
            self.dist_rbf_dim = dist_rbf
        else:
            self.dist_rbf_dim = 0
        self.num_steps = num_steps
        self.trans_x = construct_transition(trans_pos_type, num_steps, trans_pos_opt)
        self.trans_h = construct_transition(trans_seq_type, num_steps, trans_seq_opt)

        self.register_buffer('std', torch.tensor(std, dtype=torch.float))

        self.pair_init = OnlinePairInit(
            node_dim=latent_size,
            pair_dim=pair_dim,
            pair_struct_dim=pair_struct_dim,
            rbf_dim=pair_rbf_dim,
            rbf_cutoff=pair_rbf_cutoff,
            edge_type_dim=edge_type_dim,
            hidden_dim=pair_hidden_dim,
            ca_idx=ca_idx,
            anchor_mode=pair_anchor_mode,
            anchor_channel_idx=pair_anchor_channel_idx,
        )
        self.enable_covalent_edges = enable_covalent_edges
        self.covalent_edge_type_id = covalent_edge_type_id
        self.covalent_edge_start_t = covalent_edge_start_t
        self.enable_pair_injection = enable_pair_injection
        cycle_kwargs = dict(
            d_cyc=pair_dim - pair_struct_dim,
            type_hidden=type_hidden,
            pair_hidden=pair_head_hidden,
            min_sep_all=min_sep_all,
            min_sep_ss=min_sep_ss,
            allow_ht_alt_pairs=allow_ht_alt_pairs,
            ht_alt_min_end=ht_alt_min_end,
            mask_mode=mask_mode,
            relax_residue_constraints=relax_residue_constraints,
        )
        # Backward-compat: drop args not supported by older CyclizationHeads.
        sig = inspect.signature(CyclizationHeads.__init__)
        for key in list(cycle_kwargs.keys()):
            if key not in sig.parameters:
                cycle_kwargs.pop(key)
        self.cycle_head = CyclizationHeads(**cycle_kwargs)
        edge_type_vocab_size = 3
        if enable_covalent_edges:
            edge_type_vocab_size = max(edge_type_vocab_size, covalent_edge_type_id + 1)
        self.injector = EdgeFeatureInjector(
            pair_struct_dim=pair_struct_dim,
            struct_proj_dim=struct_proj_dim,
            edge_type_dim=edge_type_dim,
            edge_type_vocab_size=edge_type_vocab_size,
            covalent_edge_type_id=covalent_edge_type_id,
            separate_params=covalent_edge_separate_params,
            include_type=True,
            include_pair_score=True,
            include_cycle_edge=True,
            include_edge_type=True,
        )
        self.bias_gate_injector = BiasGateInjector(pair_struct_dim, hidden_size)
        self.lambda_cyc_max = lambda_cyc_max
        self.lambda_cyc_schedule = lambda_cyc_schedule
        self.lambda_b_max = lambda_b_max
        self.lambda_g_max = lambda_g_max

        self.pair_struct_refine = pair_struct_refine
        if pair_struct_refine:
            struct_in_dim = pair_struct_dim + pair_dim * 3 + pair_rbf_dim
            self.pair_struct_refine_mlp = nn.Sequential(
                nn.Linear(struct_in_dim, pair_struct_refine_hidden),
                nn.SiLU(),
                nn.Linear(pair_struct_refine_hidden, pair_struct_dim),
            )

        self.pair_cyc_refine = pair_cyc_refine
        if pair_cyc_refine:
            d_cyc = pair_dim - pair_struct_dim
            self.pair_cyc_refine_mlp = nn.Sequential(
                nn.Linear(pair_dim * 2 + pair_dim + pair_rbf_dim, pair_cyc_refine_hidden),
                nn.SiLU(),
                nn.Linear(pair_cyc_refine_hidden, d_cyc),
            )

        self.pocket_attn = PocketCrossAttn(latent_size, dropout=pocket_cross_attn_dropout) if enable_pocket_cross_attn else None

    def _normalize_position(self, X, batch_ids, mask_generate, atom_mask, L=None):
        base_ctx_mask = (~mask_generate[:, None].expand_as(atom_mask)) & atom_mask
        if not base_ctx_mask.any():
            raise ValueError('No context atoms available for normalization; check rec_chain/pocket context.')
        # Prefer CA (index=1) as context center; fall back to any context atom if CA is missing.
        ctx_mask = base_ctx_mask.clone()
        ctx_mask[:, 0] = 0
        ctx_mask[:, 2:] = 0
        if not ctx_mask.any():
            ctx_mask = base_ctx_mask
        centers = scatter_mean(X[ctx_mask], batch_ids[:, None].expand_as(ctx_mask)[ctx_mask], dim=0)
        centers = centers[batch_ids].unsqueeze(1)
        if L is None:
            X = (X - centers) / self.std
        else:
            with torch.no_grad():
                L_inv = low_trianguler_inv(L)
            X = X - centers
            X = torch.matmul(L_inv[batch_ids][..., None, :, :], X.unsqueeze(-1)).squeeze(-1)
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids, L=None):
        if L is None:
            X = X_norm * self.std + centers
        else:
            X = torch.matmul(L[batch_ids][..., None, :, :], X_norm.unsqueeze(-1)).squeeze(-1) + centers
        return X

    @torch.no_grad()
    def _get_batch_ids(self, mask_generate, lengths):
        batch_ids = torch.zeros_like(mask_generate).long()
        batch_ids[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_ids.cumsum_(dim=0)
        return batch_ids

    @torch.no_grad()
    def _get_edges(self, mask_generate, batch_ids, lengths):
        row, col = variadic_meshgrid(
            input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size1=lengths,
            input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
            size2=lengths,
        )
        is_ctx = mask_generate[row] == mask_generate[col]
        is_inter = ~is_ctx
        ctx_edges = torch.stack([row[is_ctx], col[is_ctx]], dim=0)
        inter_edges = torch.stack([row[is_inter], col[is_inter]], dim=0)
        return ctx_edges, inter_edges

    @torch.no_grad()
    def _get_edge_dist(self, X, edges, atom_mask):
        ca_x = X[:, 1]
        ca_x = torch.where(atom_mask[:, 1, None], X[:, 1], X[:, 0])
        dist = torch.norm(ca_x[edges[0]] - ca_x[edges[1]], dim=-1)
        return dist

    @staticmethod
    def _check_finite(label: str, value: torch.Tensor, t=None):
        if not bool(torch.isfinite(value).all()):
            bad = (~torch.isfinite(value)).nonzero(as_tuple=False)
            first = bad[:5].detach().cpu().tolist()
            raise FloatingPointError(
                f'APCyc reverse diffusion: {label} non-finite at t={t}, '
                f'count={bad.size(0)}, first indices={first}'
            )

    @contextlib.contextmanager
    def _trace_eps_layers(self, t, X_t, H_t, atom_mask, mask_generate, batch_ids):
        """Optionally pinpoint the first non-finite EGNN intermediate.

        Enable with APCYC_TRACE_EPS=1. Hooks are temporary and always removed.
        This is a diagnostic only: no values are clamped or replaced.
        """
        if os.environ.get('APCYC_TRACE_EPS') != '1':
            yield
            return

        def check(label, obj):
            if torch.is_tensor(obj):
                if not (obj.is_floating_point() or obj.is_complex()):
                    return
                bad = ~torch.isfinite(obj)
                if bool(bad.any()):
                    first = bad.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
                    raise FloatingPointError(
                        f'APCyc EpsilonNet first invalid {label}, t={t}, '
                        f'shape={tuple(obj.shape)}, dtype={obj.dtype}, '
                        f'bad={int(bad.sum().item())}, first={first}'
                    )
            elif isinstance(obj, (tuple, list)):
                for i, v in enumerate(obj):
                    check(f'{label}[{i}]', v)
            elif isinstance(obj, dict):
                for key, v in obj.items():
                    check(f'{label}.{key}', v)

        def summary(label, tensor):
            finite = tensor[torch.isfinite(tensor)]
            magnitude = float(finite.abs().max().item()) if finite.numel() else float('nan')
            return f'{label}.max_abs={magnitude:.6g}'

        # coord2radial() divides by the number of active channels of each node.
        # A zero here is a real bad input, not something to silently clamp away.
        counts = (atom_mask != 0).sum(dim=1)
        missing = (counts == 0).nonzero(as_tuple=False).flatten()
        if missing.numel():
            first = missing[:5]
            particles = batch_ids[first].detach().cpu().tolist()
            raise FloatingPointError(
                f'APCyc EpsilonNet: zero active atom channels at t={t}, '
                f'local nodes={first.detach().cpu().tolist()}, local particles={particles}'
            )

        print('[APCyc EpsTrace]', f't={t}',
              summary('X_t', X_t), summary('H_t', H_t),
              f'generated_nodes={int(mask_generate.sum().item())}',
              f'local_particles={int(batch_ids.max().item()) + 1}',
              flush=True)
        hooks = []

        def pre_hook(label):
            def hook(_module, inputs):
                check(label + '.input', inputs)
            return hook

        def post_hook(label):
            def hook(_module, _inputs, output):
                check(label + '.output', output)
            return hook

        # Check full GCL outputs, but also check submodules to distinguish
        # radial/message overflow from unstable coordinate updates.
        for name, layer in self.eps_net.encoder.named_modules():
            track = (name == 'linear_in' or name == 'linear_out' or
                     name == 'out_layer' or
                     (name.startswith('gcl_') and '.' not in name) or
                     name.endswith(('.edge_mlp', '.coord_mlp', '.node_mlp',
                                    '.radial_linear', '.rbf_linear')) or
                     name in ('out_layer.edge_mlp', 'out_layer.coord_mlp',
                              'out_layer.node_mlp', 'out_layer.radial_linear',
                              'out_layer.rbf_linear'))
            if track:
                hooks.append(layer.register_forward_pre_hook(pre_hook(name)))
                hooks.append(layer.register_forward_hook(post_hook(name)))
        try:
            yield
        finally:
            for handle in hooks:
                handle.remove()

    def _get_pep_indices(self, mask_generate, batch_ids) -> List[torch.Tensor]:
        pep_indices = []
        bs = int(batch_ids.max().item()) + 1
        for b in range(bs):
            idx = torch.nonzero((batch_ids == b) & mask_generate).view(-1)
            pep_indices.append(idx)
        return pep_indices

    def _pair_forward(
        self,
        H_noisy: torch.Tensor,
        X_noisy: torch.Tensor,
        S: torch.Tensor,
        mask_generate: torch.Tensor,
        batch_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        type_labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self._check_finite('pair input H_noisy', H_noisy)
        self._check_finite('pair input X_noisy', X_noisy)
        pair_base, pair_struct, pair_cyc, dist = self.pair_init(
            H_noisy, X_noisy, mask_generate, batch_ids=batch_ids, position_ids=position_ids
        )
        self._check_finite('pair_init.pair_base', pair_base)
        self._check_finite('pair_init.pair_struct', pair_struct)
        self._check_finite('pair_init.pair_cyc', pair_cyc)
        self._check_finite('pair_init.dist', dist)
        if self.pair_struct_refine or self.pair_cyc_refine:
            phi = self.pair_init.node_ln(self.pair_init.node_proj(H_noisy))
            n_nodes = phi.size(0)
            phi_i = phi[:, None, :].expand(n_nodes, n_nodes, -1)
            phi_j = phi[None, :, :].expand(n_nodes, n_nodes, -1)
            phi_mul = phi_i * phi_j
            rbf = self.pair_init.rbf(dist.view(-1)).view(dist.size(0), dist.size(1), -1)
        if self.pair_struct_refine:
            struct_refine_in = torch.cat([pair_struct, phi_i, phi_j, phi_mul, rbf], dim=-1)
            pair_struct = pair_struct + self.pair_struct_refine_mlp(struct_refine_in)
        if self.pair_cyc_refine:
            cyc_refine_in = torch.cat([phi_i, phi_j, phi_mul, rbf], dim=-1)
            pair_cyc = pair_cyc + self.pair_cyc_refine_mlp(cyc_refine_in)
        pep_indices = self._get_pep_indices(mask_generate, batch_ids)
        self._check_finite('refined pair_struct', pair_struct)
        self._check_finite('refined pair_cyc', pair_cyc)
        pep_indices = self._get_pep_indices(mask_generate, batch_ids)
        head_out = self.cycle_head(pair_cyc, pep_indices, S, type_labels=type_labels)
        head_out.update({
            'pair_struct': pair_struct,
            'pair_cyc': pair_cyc,
            'pair_dist': dist,
            'pep_indices': pep_indices,
        })
        return head_out

    def _build_edge_type(self, part_mask: torch.Tensor, hard_pair: Optional[torch.Tensor]) -> torch.Tensor:
        edge_type = (part_mask[:, None] != part_mask[None, :]).long()
        if hard_pair is None:
            return edge_type
        for b in range(hard_pair.size(0)):
            i, j = hard_pair[b].tolist()
            if i < 0 or j < 0 or i == j:
                continue
            edge_type[i, j] = 2
            edge_type[j, i] = 2
        return edge_type

    def _build_covalent_edges(
        self,
        hard_pairs: torch.Tensor,
        enable_mask: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        rows, cols, types = [], [], []
        for b in range(hard_pairs.size(0)):
            if not bool(enable_mask[b].item()):
                continue
            i, j = hard_pairs[b].tolist()
            if i < 0 or j < 0 or i == j:
                continue
            rows.extend([i, j])
            cols.extend([j, i])
            types.extend([self.covalent_edge_type_id, self.covalent_edge_type_id])
        if not rows:
            return None
        edges = torch.tensor([rows, cols], device=hard_pairs.device, dtype=torch.long)
        edge_types = torch.tensor(types, device=hard_pairs.device, dtype=torch.long)
        return edges, edge_types

    @staticmethod
    def _append_covalent_edges(ctx_edges, inter_edges, edge_type_ids, cov_pack):
        """Keep edge data in EXACTLY [context(+covalent), inter] order."""
        if cov_pack is None:
            return ctx_edges, torch.cat([ctx_edges, inter_edges], dim=-1), edge_type_ids
        cov_edges, cov_types = cov_pack
        n_ctx = ctx_edges.shape[1]
        ctx_edges = torch.cat([ctx_edges, cov_edges], dim=-1)
        reordered_types = torch.cat([
            edge_type_ids[:n_ctx], cov_types, edge_type_ids[n_ctx:]
        ], dim=0)
        return ctx_edges, torch.cat([ctx_edges, inter_edges], dim=-1), reordered_types

    def _pair_dist_from_X(self, X: torch.Tensor, batch_ids: Optional[torch.Tensor]) -> torch.Tensor:
        anchor = get_anchor_coords(
            X,
            mode=self.pair_init.anchor_mode,
            ca_idx=self.pair_init.ca_idx,
            channel_idx=self.pair_init.anchor_channel_idx,
        )
        dist = torch.cdist(anchor, anchor, p=2)
        if batch_ids is not None:
            same_batch = (batch_ids[:, None] == batch_ids[None, :]).to(dist.dtype)
            dist = dist * same_batch
        return dist

    def _predict_x0(self, X_noisy: torch.Tensor, eps_X_pred: torch.Tensor, batch_ids: torch.Tensor, t: torch.Tensor) -> Optional[torch.Tensor]:
        if not hasattr(self.trans_x, 'var_sched'):
            return None
        expand_shape = [X_noisy.shape[0]] + [1 for _ in X_noisy.shape[1:]]
        alpha_bar = self.trans_x.var_sched.alpha_bars[t][batch_ids]
        c0 = torch.sqrt(alpha_bar).view(*expand_shape)
        c1 = torch.sqrt(1 - alpha_bar).view(*expand_shape)
        return (X_noisy - c1 * eps_X_pred) / (c0 + 1e-8)

    def _hard_pair_s_ij(self, s_ij: torch.Tensor, pep_indices: List[torch.Tensor]) -> torch.Tensor:
        s_hard = torch.zeros_like(s_ij)
        for pep_idx in pep_indices:
            if pep_idx.numel() < 2:
                continue
            s_pp = s_ij[pep_idx][:, pep_idx]
            if not bool(torch.isfinite(s_pp).all()):
                raise FloatingPointError('Non-finite hard-pair scores')
            if not bool((s_pp > 0).any()):
                continue  # All candidates masked; do not invent a bond.
            flat_idx = torch.argmax(s_pp.reshape(-1))
            i = int(flat_idx // s_pp.size(1))
            j = int(flat_idx % s_pp.size(1))
            gi, gj = pep_idx[i], pep_idx[j]
            s_hard[gi, gj] = 1.0
            s_hard[gj, gi] = 1.0
        return s_hard

    def forward(
        self,
        H_0,
        X_0,
        position_embedding,
        mask_generate,
        lengths,
        atom_embeddings,
        atom_mask,
        S,
        position_ids=None,
        L=None,
        t=None,
        sample_structure=True,
        sample_sequence=True,
        type_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        batch_size = batch_ids.max() + 1
        if t is None:
            t = torch.randint(0, self.num_steps + 1, (batch_size,), dtype=torch.long, device=H_0.device)

        X_0, centers = self._normalize_position(X_0, batch_ids, mask_generate, atom_mask, L)
        if L is not None and APCYC_VERBOSE:
            singular_values = torch.linalg.svdvals(L.double())

            print(
                "[APCyc covariance]",
                "min_singular=",
                singular_values.min().item(),
                "max_singular=",
                singular_values.max().item(),
                "max_condition=",
                (
                    singular_values.max(dim=-1).values /
                    singular_values.min(dim=-1).values.clamp_min(1e-15)
                ).max().item(),
                flush=True,
            )

        if sample_structure:
            X_noisy, eps_X = self.trans_x.add_noise(X_0, mask_generate, batch_ids, t)
        else:
            X_noisy, eps_X = X_0, torch.zeros_like(X_0)
        if sample_sequence:
            H_noisy, eps_H = self.trans_h.add_noise(H_0, mask_generate, batch_ids, t)
        else:
            H_noisy, eps_H = H_0, torch.zeros_like(H_0)

        if self.pocket_attn is not None:
            H_noisy = self.pocket_attn(H_noisy, mask_generate, batch_ids)

        ctx_edges, inter_edges = self._get_edges(mask_generate, batch_ids, lengths)
        edges = torch.cat([ctx_edges, inter_edges], dim=-1)

        X_noisy_unorm = self._unnormalize_position(X_noisy, centers, batch_ids, L)
        pair_out = self._pair_forward(
            H_noisy, X_noisy_unorm, S, mask_generate, batch_ids, position_ids, type_labels
        )

        self._check_finite('training pair probabilities', pair_out['s_ij_full'], t)
        self._check_finite('training pair structure', pair_out['pair_struct'], t)
        p_c = torch.softmax(pair_out['type_logits'].float(), dim=-1).to(pair_out['type_logits'].dtype)
        lambda_cyc = cyclic_lambda(t, self.num_steps, self.lambda_cyc_max, self.lambda_cyc_schedule)
        lambda_b = cyclic_lambda(t, self.num_steps, self.lambda_b_max, self.lambda_cyc_schedule)
        lambda_g = cyclic_lambda(t, self.num_steps, self.lambda_g_max, self.lambda_cyc_schedule)

        s_ij_inject = pair_out['s_ij_full'] + pair_out['s_ij_full'].transpose(0, 1)
        edge_type = self._build_edge_type(mask_generate, hard_pair=None)
        edge_type_ids = edge_type[edges[0], edges[1]]

        if self.enable_pair_injection:
            hard_pairs = None
            if self.enable_covalent_edges and self.covalent_edge_start_t is not None:
                enable_mask = (t <= self.covalent_edge_start_t)
                if enable_mask.any():
                    hard_pairs = self._argmax_pairs(pair_out['s_ij_full'], pair_out['pep_indices'])
                    cov_pack = self._build_covalent_edges(hard_pairs, enable_mask)
                    if cov_pack is not None:
                        ctx_edges_g, edges_g, edge_type_ids = self._append_covalent_edges(ctx_edges_g, inter_edges_g, edge_type_ids, cov_pack)

            pair_edge_attr = self.injector(
                edges,
                batch_ids,
                pair_out['pair_struct'],
                s_ij_inject,
                p_c,
                lambda_cyc,
                edge_type=edge_type_ids,
                hard_pair=hard_pairs,
            )
            ctx_pair_attr = pair_edge_attr[: ctx_edges.shape[1]]
            inter_pair_attr = pair_edge_attr[ctx_edges.shape[1] :]

            edge_bias, edge_gate = self.bias_gate_injector(
                edges,
                batch_ids,
                pair_out['pair_struct'],
                s_ij_inject,
                lambda_b,
                lambda_g,
            )
        else:
            pair_edge_attr = pair_out['pair_struct'].new_zeros(edges.size(1), self.pair_edge_dim)
            ctx_pair_attr = pair_edge_attr[: ctx_edges.shape[1]]
            inter_pair_attr = pair_edge_attr[ctx_edges.shape[1] :]
            edge_bias, edge_gate = None, None

        if hasattr(self, 'dist_rbf'):
            ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), ctx_edges, atom_mask)
            inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_noisy, centers, batch_ids, L), inter_edges, atom_mask)
            ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
            inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
            ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
            inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
        else:
            ctx_edge_attr, inter_edge_attr = ctx_pair_attr, inter_pair_attr

        beta = self.trans_x.get_timestamp(t)[batch_ids]
        eps_H_pred, eps_X_pred = self.eps_net(
            H_noisy, X_noisy, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,
            ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,
            edge_bias=edge_bias, edge_gate=edge_gate)

        X_pred = self._predict_x0(X_noisy, eps_X_pred, batch_ids, t)
        if X_pred is not None:
            X_pred_unorm = self._unnormalize_position(X_pred, centers, batch_ids, L)
            pair_out['pair_dist_pred'] = self._pair_dist_from_X(X_pred_unorm, batch_ids)

        loss_dict = {}
        if sample_structure:
            mask_loss = mask_generate[:, None] & atom_mask
            loss_X = F.mse_loss(eps_X_pred[mask_loss], eps_X[mask_loss], reduction='none').sum(dim=-1)
            loss_X = loss_X.sum() / (mask_loss.sum().float() + 1e-8)
            loss_dict['X'] = loss_X
        else:
            loss_dict['X'] = 0

        if sample_sequence:
            loss_H = F.mse_loss(eps_H_pred[mask_generate], eps_H[mask_generate], reduction='none').sum(dim=-1)
            loss_H = loss_H.sum() / (mask_generate.sum().float() + 1e-8)
            loss_dict['H'] = loss_H
        else:
            loss_dict['H'] = 0

        return loss_dict, pair_out

    @torch.no_grad()
    def sample(
        self,
        H,
        X,
        position_embedding,
        mask_generate,
        lengths,
        atom_embeddings,
        atom_mask,
        S,
        position_ids=None,
        L=None,
        sample_structure=True,
        sample_sequence=True,
        pbar=False,
        energy_func=None,
        energy_lambda=0.01,
        t_hard: int = 0,
        return_pair_trajectory: bool = False,
        use_cg_guidance: bool = False,
        property_predictor=None,
        guidance_w=None,
        guidance_w_min: float = None,
        guidance_w_max: float = None,
        guidance_grad_norm_balance: bool = False,
        guidance_grad_norm_eps: float = 1e-6,
        guidance_grad_norm_ref: str = 'mean',
        lambda_max: float = 0.0,
        t_guidance_start: int = 0,
        t_guidance_end: int = 0,
        smc_state=None,
        smc_initial_state=None,
        smc_start_t=None,
        smc_stop_t=1,
    ):
        batch_ids = self._get_batch_ids(mask_generate, lengths)
        X, centers = self._normalize_position(X, batch_ids, mask_generate, atom_mask, L)

        if sample_structure:
            X_rand = torch.randn_like(X)
            X_init = torch.where(mask_generate[:, None, None].expand_as(X), X_rand, X)
        else:
            X_init = X

        if sample_sequence:
            H_rand = torch.randn_like(H)
            H_init = torch.where(mask_generate[:, None].expand_as(H), H_rand, H)
        else:
            H_init = H

        # resume from normalized latents for a short time window without changing physical receptor/context coordinates.
        smc_first_t = self.num_steps if smc_start_t is None else int(smc_start_t)
        smc_last_t = int(smc_stop_t)

        if not 1 <= smc_last_t <= smc_first_t <= self.num_steps:
            raise ValueError('Invalid global SMC diffusion window')
        if smc_initial_state is not None:
            X_init, H_init = smc_initial_state
            if X_init.shape != X.shape or H_init.shape != H.shape:
                raise ValueError('Global SMC initial latent shape mismatch')
            if X_init.device != X.device or H_init.device != H.device:
                raise ValueError('Global SMC initial latents must use sampler device')
        elif smc_start_t is not None and smc_first_t != self.num_steps:
            raise ValueError('Global SMC continuation requires previous latent state')
        
        self._check_finite('X_init', X_init, smc_first_t)
        self._check_finite('H_init', H_init, smc_first_t)
        traj = {smc_first_t: (X_init, H_init)}
        pair_traj = []
        hard_pairs = None

        guidance_w_tensor = None
        if use_cg_guidance and property_predictor is not None:
            guidance_w_tensor = parse_guidance_w(
                guidance_w, property_predictor.n_properties, device=H.device
            )
            guidance_w_tensor = clip_guidance_w(guidance_w_tensor, guidance_w_min, guidance_w_max)

        if pbar:
            pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        else:
            pbar = lambda x: x

        for t in pbar(range(smc_first_t, smc_last_t - 1, -1)):
            X_t, H_t = traj[t]
            check_particle_scale("X_t entering step", X_t, t, batch_ids, mask_generate)

            X_t, H_t = torch.round(X_t, decimals=4), torch.round(H_t, decimals=4)
            self._check_finite('X_t', X_t, t)
            self._check_finite('H_t', H_t, t)

            beta = self.trans_x.get_timestamp(t).view(1).repeat(X_t.shape[0])
            t_tensor = torch.full([X_t.shape[0], ], fill_value=t, dtype=torch.long, device=X_t.device)

            if self.pocket_attn is not None:
                H_t = self.pocket_attn(H_t, mask_generate, batch_ids)
            self._check_finite('H_t after pocket attention', H_t, t)
            H_step_start = H_t

            ctx_edges, inter_edges = self._get_edges(mask_generate, batch_ids, lengths)
            edges = torch.cat([ctx_edges, inter_edges], dim=-1)

            X_t_unorm = self._unnormalize_position(X_t, centers, batch_ids, L)
            self._check_finite('X_t unnormalized for pair head', X_t_unorm, t)
            try:
                pair_out = self._pair_forward(
                    H_t, X_t_unorm, S, mask_generate, batch_ids, position_ids, type_labels=None
                )
            except FloatingPointError as exc:
                raise FloatingPointError(
                    f'APCyc pair construction failed at reverse timestep t={t}: {exc}'
                ) from exc
            if smc_state is not None:
                smc_state.repair_pair_probabilities(t, pair_out)
            self._check_finite('pair_struct', pair_out['pair_struct'], t)
            self._check_finite('s_ij_full', pair_out['s_ij_full'], t)
            p_c = torch.softmax(pair_out['type_logits'].float(), dim=-1).to(pair_out['type_logits'].dtype)
            self._check_finite('p_c', p_c, t)
            t_batch = torch.full((p_c.size(0),), t, device=H_t.device, dtype=torch.long)
            lambda_cyc = cyclic_lambda(t_batch, self.num_steps, self.lambda_cyc_max, self.lambda_cyc_schedule)
            lambda_b = cyclic_lambda(t_batch, self.num_steps, self.lambda_b_max, self.lambda_cyc_schedule)
            lambda_g = cyclic_lambda(t_batch, self.num_steps, self.lambda_g_max, self.lambda_cyc_schedule)

            hard_pairs = None
            s_ij_inject = pair_out['s_ij_full'] + pair_out['s_ij_full'].transpose(0, 1)
            if t <= t_hard:
                hard_pairs = self._argmax_pairs(pair_out['s_ij_full'], pair_out['pep_indices'])
                s_ij_inject = self._hard_pair_s_ij(s_ij_inject, pair_out['pep_indices'])

            edge_type = self._build_edge_type(mask_generate, hard_pairs)
            edge_type_ids = edge_type[edges[0], edges[1]]

            if self.enable_pair_injection:
                if self.enable_covalent_edges:
                    start_t = self.covalent_edge_start_t if self.covalent_edge_start_t is not None else t_hard
                    if t <= start_t:
                        if hard_pairs is None:
                            hard_pairs = self._argmax_pairs(pair_out['s_ij_full'], pair_out['pep_indices'])
                        cov_pack = self._build_covalent_edges(hard_pairs, torch.ones_like(p_c[:, 0], dtype=torch.bool))
                        if cov_pack is not None:
                            ctx_edges, edges, edge_type_ids = self._append_covalent_edges(ctx_edges, inter_edges, edge_type_ids, cov_pack)
                pair_edge_attr = self.injector(
                    edges,
                    batch_ids,
                    pair_out['pair_struct'],
                    s_ij_inject,
                    p_c,
                    lambda_cyc,
                    edge_type=edge_type_ids,
                    hard_pair=hard_pairs,
                )
                ctx_pair_attr = pair_edge_attr[: ctx_edges.shape[1]]
                inter_pair_attr = pair_edge_attr[ctx_edges.shape[1] :]

                edge_bias, edge_gate = self.bias_gate_injector(
                    edges,
                    batch_ids,
                    pair_out['pair_struct'],
                    s_ij_inject,
                    lambda_b,
                    lambda_g,
                )
            else:
                pair_edge_attr = pair_out['pair_struct'].new_zeros(edges.size(1), self.pair_edge_dim)
                ctx_pair_attr = pair_edge_attr[: ctx_edges.shape[1]]
                inter_pair_attr = pair_edge_attr[ctx_edges.shape[1] :]
                edge_bias, edge_gate = None, None

            if hasattr(self, 'dist_rbf'):
                ctx_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), ctx_edges, atom_mask)
                inter_edge_attr = self._get_edge_dist(self._unnormalize_position(X_t, centers, batch_ids, L), inter_edges, atom_mask)
                ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges.shape[1], -1)
                inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges.shape[1], -1)
                ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
                inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
            else:
                ctx_edge_attr, inter_edge_attr = ctx_pair_attr, inter_pair_attr

            self._check_finite('ctx_edge_attr', ctx_edge_attr, t)
            self._check_finite('inter_edge_attr', inter_edge_attr, t)
            if edge_bias is not None:
                self._check_finite('edge_bias', edge_bias, t)
                self._check_finite('edge_gate', edge_gate, t)
            with self._trace_eps_layers(t, X_t, H_t, atom_mask, mask_generate, batch_ids):
                eps_H, eps_X = self.eps_net(
                    H_t, X_t, position_embedding, ctx_edges, inter_edges, atom_embeddings, atom_mask.float(), mask_generate, beta,
                    ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,
                    edge_bias=edge_bias, edge_gate=edge_gate)
            for label, prediction in (('eps_H', eps_H), ('eps_X', eps_X)):
                try:
                    self._check_finite(label, prediction, t)
                except FloatingPointError as exc:
                    per_node_bad = (~torch.isfinite(prediction)).reshape(prediction.shape[0], -1).any(dim=1)
                    bad_particles = torch.unique(batch_ids[per_node_bad]).detach().cpu().tolist()
                    bad_nodes = per_node_bad.nonzero(as_tuple=False).flatten()[:5].detach().cpu().tolist()
                    raise FloatingPointError(
                        f'{exc}; local_particles={bad_particles}, local_nodes={bad_nodes}; '
                        'rerun with APCYC_TRACE_EPS=1 for first failing EGNN module'
                    ) from exc

            if smc_state is not None and smc_state.should_score(t):
                smc_state.observe(
                    t, pair_out=pair_out, predictor=property_predictor,
                    X_unorm=X_t_unorm,
                    H_t=H_t, X_t=X_t, mask_generate=mask_generate,
                    atom_mask=atom_mask, position_ids=position_ids,
                    ctx_edges=ctx_edges, inter_edges=inter_edges,
                    ctx_edge_attr=ctx_edge_attr, inter_edge_attr=inter_edge_attr,
                    edge_gate=edge_gate, edge_bias=edge_bias,
                    edge_type_ids=edge_type_ids, p_c=p_c, t_batch=t_batch,
                    batch_ids=batch_ids, atom_embeddings=atom_embeddings)

            if energy_func is not None:
                with torch.enable_grad():
                    cur_X_state = X_t.clone().double()
                    cur_X_state.requires_grad = True
                    if L is None:
                        raise ValueError('Energy guidance requires covariance matrix L (use_covariance_matrix=true).')
                    energy = energy_func(
                        X=self._unnormalize_position(cur_X_state, centers.double(), batch_ids, L.double()),
                        mask_generate=mask_generate, batch_ids=batch_ids)
                    energy_eps_X = grad([energy], [cur_X_state], create_graph=False, retain_graph=False)[0].float()
                energy_eps_X[~mask_generate] = 0
                energy_eps_X = -energy_eps_X
                self._check_finite('energy_guidance', energy_eps_X, t)
            else:
                energy_eps_X = None

            if APCYC_VERBOSE:
                trace_magnitude("X_t", X_t, t, mask_generate)
                trace_magnitude("eps_X", eps_X, t, mask_generate)

            H_next = self.trans_h.denoise(H_t, eps_H, mask_generate, batch_ids, t_tensor)
            X_next = self.trans_x.denoise(X_t, eps_X, mask_generate, batch_ids, t_tensor, guidance=energy_eps_X, guidance_weight=energy_lambda)
            if APCYC_VERBOSE:
                trace_magnitude("X_next", X_next, t, mask_generate)
            check_particle_scale("X_next after denoise", X_next, t, batch_ids, mask_generate)

            if use_cg_guidance and property_predictor is not None and t <= t_guidance_start:
                H_guid = H_t.detach().requires_grad_(True)
                X_guid = X_t.detach().requires_grad_(True)
                ctx_edges_g, inter_edges_g = self._get_edges(mask_generate, batch_ids, lengths)
                edges_g = torch.cat([ctx_edges_g, inter_edges_g], dim=-1)
                X_guid_unorm = self._unnormalize_position(X_guid, centers, batch_ids, L)
                pair_out_guid = self._pair_forward(
                    H_guid, X_guid_unorm, S, mask_generate, batch_ids, position_ids, type_labels=None
                )
                self._check_finite('guided s_ij_full', pair_out_guid['s_ij_full'], t)
                self._check_finite('guided pair_struct', pair_out_guid['pair_struct'], t)
                p_c_guid = torch.softmax(pair_out_guid['type_logits'].float(), dim=-1).to(pair_out_guid['type_logits'].dtype)
                self._check_finite('guided p_c', p_c_guid, t)
                t_batch = torch.full((p_c_guid.size(0),), t, device=H_t.device, dtype=torch.long)
                lambda_cyc = cyclic_lambda(t_batch, self.num_steps, self.lambda_cyc_max, self.lambda_cyc_schedule)
                lambda_b = cyclic_lambda(t_batch, self.num_steps, self.lambda_b_max, self.lambda_cyc_schedule)
                lambda_g = cyclic_lambda(t_batch, self.num_steps, self.lambda_g_max, self.lambda_cyc_schedule)

                hard_pairs = None
                s_ij_inject = pair_out_guid['s_ij_full'] + pair_out_guid['s_ij_full'].transpose(0, 1)
                if t <= t_hard:
                    hard_pairs = self._argmax_pairs(pair_out_guid['s_ij_full'], pair_out_guid['pep_indices'])
                    s_ij_inject = self._hard_pair_s_ij(s_ij_inject, pair_out_guid['pep_indices'])

                edge_type = self._build_edge_type(mask_generate, hard_pairs)
                edge_type_ids = edge_type[edges_g[0], edges_g[1]]

                if self.enable_pair_injection:
                    if self.enable_covalent_edges:
                        start_t = self.covalent_edge_start_t if self.covalent_edge_start_t is not None else t_hard
                        if t <= start_t:
                            if hard_pairs is None:
                                hard_pairs = self._argmax_pairs(pair_out_guid['s_ij_full'], pair_out_guid['pep_indices'])
                            cov_pack = self._build_covalent_edges(hard_pairs, torch.ones_like(p_c_guid[:, 0], dtype=torch.bool))
                            if cov_pack is not None:
                                ctx_edges, edges, edge_type_ids = self._append_covalent_edges(ctx_edges, inter_edges, edge_type_ids, cov_pack)

                    pair_edge_attr = self.injector(
                        edges_g,
                        batch_ids,
                        pair_out_guid['pair_struct'],
                        s_ij_inject,
                        p_c_guid,
                        lambda_cyc,
                        edge_type=edge_type_ids,
                        hard_pair=hard_pairs,
                    )
                    ctx_pair_attr = pair_edge_attr[: ctx_edges_g.shape[1]]
                    inter_pair_attr = pair_edge_attr[ctx_edges_g.shape[1] :]

                    edge_bias, edge_gate = self.bias_gate_injector(
                        edges_g,
                        batch_ids,
                        pair_out_guid['pair_struct'],
                        s_ij_inject,
                        lambda_b,
                        lambda_g,
                    )
                else:
                    pair_edge_attr = pair_out_guid['pair_struct'].new_zeros(edges_g.size(1), self.pair_edge_dim)
                    ctx_pair_attr = pair_edge_attr[: ctx_edges_g.shape[1]]
                    inter_pair_attr = pair_edge_attr[ctx_edges_g.shape[1] :]
                    edge_bias, edge_gate = None, None

                if hasattr(self, 'dist_rbf'):
                    ctx_edge_attr = self._get_edge_dist(X_guid_unorm, ctx_edges_g, atom_mask)
                    inter_edge_attr = self._get_edge_dist(X_guid_unorm, inter_edges_g, atom_mask)
                    ctx_edge_attr = self.dist_rbf(ctx_edge_attr).view(ctx_edges_g.shape[1], -1)
                    inter_edge_attr = self.dist_rbf(inter_edge_attr).view(inter_edges_g.shape[1], -1)
                    ctx_edge_attr = torch.cat([ctx_edge_attr, ctx_pair_attr], dim=-1)
                    inter_edge_attr = torch.cat([inter_edge_attr, inter_pair_attr], dim=-1)
                else:
                    ctx_edge_attr, inter_edge_attr = ctx_pair_attr, inter_pair_attr

                edge_attr = torch.cat([ctx_edge_attr, inter_edge_attr], dim=0)
                y_hat = property_predictor(
                    H_guid,
                    X_guid,
                    mask_generate,
                    atom_mask,
                    position_ids,
                    ctx_edges_g,
                    inter_edges_g,
                    edge_attr=edge_attr,
                    edge_gate=edge_gate,
                    edge_bias=edge_bias,
                    edge_type=edge_type_ids,
                    pair_struct=pair_out_guid['pair_struct'],
                    pair_cyc=pair_out_guid['pair_cyc'],
                    type_probs=p_c_guid,
                    s_ij=pair_out_guid['s_ij_full'],
                    t=t_batch,
                    batch_ids=batch_ids,
                    atom_embeddings=atom_embeddings,
                )
                if guidance_grad_norm_balance:
                    g_H_list = []
                    g_X_list = []
                    norms = []
                    for k in range(y_hat.size(-1)):
                        if guidance_w_tensor is not None and guidance_w_tensor[k].item() == 0:
                            g_H_list.append(torch.zeros_like(H_guid))
                            g_X_list.append(torch.zeros_like(X_guid))
                            norms.append(torch.zeros((), device=H_guid.device))
                            continue
                        energy_k = -y_hat[:, k].mean()
                        g_H_k, g_X_k = torch.autograd.grad(
                            energy_k,
                            [H_guid, X_guid],
                            create_graph=False,
                            retain_graph=True,
                        )
                        g_H_k = g_H_k.clone()
                        g_X_k = g_X_k.clone()
                        g_H_k[~mask_generate] = 0
                        g_X_k[~mask_generate] = 0
                        norm_k = torch.sqrt(g_H_k.pow(2).mean() + g_X_k.pow(2).mean())
                        g_H_list.append(g_H_k)
                        g_X_list.append(g_X_k)
                        norms.append(norm_k)
                    norms = torch.stack(norms)
                    w_eff = grad_norm_balance(
                        guidance_w_tensor,
                        norms,
                        eps=guidance_grad_norm_eps,
                        ref=guidance_grad_norm_ref,
                    )
                    g_H = torch.zeros_like(H_guid)
                    g_X = torch.zeros_like(X_guid)
                    for k in range(len(g_H_list)):
                        g_H = g_H + w_eff[k] * g_H_list[k]
                        g_X = g_X + w_eff[k] * g_X_list[k]
                else:
                    energy = energy_from_pred(y_hat, guidance_w_tensor)
                    g_H, g_X = torch.autograd.grad(energy, [H_guid, X_guid], create_graph=False, retain_graph=False)
                    g_H = g_H.clone()
                    g_X = g_X.clone()
                    g_H[~mask_generate] = 0
                    g_X[~mask_generate] = 0

                s_H = sigma2_from_transition(self.trans_h, t, batch_ids, (H_t.shape[0], 1))
                s_X = sigma2_from_transition(self.trans_x, t, batch_ids, (X_t.shape[0], 1, 1))
                lambda_t = guidance_lambda(t, t_guidance_start, t_guidance_end, lambda_max)
                H_next = H_next - lambda_t * s_H * g_H
                X_next = X_next - lambda_t * s_X * g_X

            if not sample_structure:
                X_next = X_t
            if not sample_sequence:
                H_next = H_t

            # Limit per-residue H update before the new state enters SMC.
            if sample_sequence:
                if max_h_step > 0:
                    H_next = clip_h_update(
                        H_step_start,
                        H_next,
                        mask_generate,
                        max_norm=max_h_step,
                    )
            
            self._check_finite('X_next before SMC', X_next, t)
            self._check_finite('H_next before SMC', H_next, t)
            if smc_state is not None:
                X_next, H_next = smc_state.resample_step(t, X_next, H_next)
            self._check_finite('X_next after SMC', X_next, t)
            self._check_finite('H_next after SMC', H_next, t)
            check_particle_scale("X_next final", X_next, t, batch_ids, mask_generate)
            traj[t - 1] = (X_next, H_next)
            if smc_start_t is None:
                # trajectory behavior for baseline runs.
                traj[t] = (self._unnormalize_position(traj[t][0], centers, batch_ids, L).cpu(), traj[t][1].cpu())
            else:
                # The global runner keeps just the current normalized state, avoiding a CPU/GPU trajectory history for every particle.
                del traj[t]

            if return_pair_trajectory:
                pair_traj.append({
                    't': t,
                    'type_logits': pair_out['type_logits'].detach().cpu(),
                    's_ij_full': pair_out['s_ij_full'].detach().cpu(),
                })

        if smc_start_t is not None and smc_last_t > 1:
            return {smc_last_t - 1: traj[smc_last_t - 1]}, None

        X_norm = traj[0][0]
        X_phys = self._unnormalize_position(X_norm, centers, batch_ids, L)

        if APCYC_VERBOSE:
            active = mask_generate[:, None] & atom_mask.bool()
            inactive = mask_generate[:, None] & ~atom_mask.bool()

            print(
                "[APCyc final diffusion]",
                "normalized_active_max=",
                X_norm[active].abs().max().item(),
                "physical_active_max=",
                X_phys[active].abs().max().item(),
                "physical_inactive_max=",
                X_phys[inactive].abs().max().item()
                if inactive.any() else 0.0,
                flush=True,
            )

            # Distance of active generated channels from
            # their receptor-context coordinate center.
            offset = torch.linalg.vector_norm(
                X_phys - centers, dim=-1
            )

            print(
                "[APCyc geometry]",
                "max_center_distance=",
                offset[active].max().item(),
                flush=True,
            )
        
        traj[0] = (self._unnormalize_position(traj[0][0], centers, batch_ids, L), traj[0][1])
        self._check_finite('final decoded coordinate latents', traj[0][0], 0)
        self._check_finite('final sequence latents', traj[0][1], 0)

        pair_state = {
            'type_logits': pair_out['type_logits'],
            's_ij_full': pair_out['s_ij_full'],
            'pep_indices': pair_out['pep_indices'],
            'pair_traj': pair_traj if return_pair_trajectory else None,
        }
        return traj, pair_state

    def _argmax_pairs(self, s_ij_full: torch.Tensor, pep_indices: List[torch.Tensor]) -> torch.Tensor:
        pairs = []
        for pep_idx in pep_indices:
            if pep_idx.numel() < 2:
                pairs.append(torch.tensor([-1, -1], device=s_ij_full.device))
                continue
            s_pp = s_ij_full[pep_idx][:, pep_idx]
            if not bool(torch.isfinite(s_pp).all()):
                raise FloatingPointError('Non-finite cyclization pair scores')
            if not bool((s_pp > 0).any()):
                pairs.append(torch.tensor([-1, -1], device=s_ij_full.device))
                continue
            flat_idx = torch.argmax(s_pp.reshape(-1))
            i = int(flat_idx // s_pp.size(1))
            j = int(flat_idx % s_pp.size(1))
            pairs.append(torch.tensor([pep_idx[i], pep_idx[j]], device=s_ij_full.device))
        return torch.stack(pairs, dim=0)
