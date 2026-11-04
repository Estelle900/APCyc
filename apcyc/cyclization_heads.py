#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from data.format import VOCAB
from .constants import CYCLE_TYPES, TYPE2IDX


def _resolve_symbol(symbol: str) -> str:
    if hasattr(VOCAB, 'resolve_symbol'):
        return VOCAB.resolve_symbol(symbol)
    if hasattr(VOCAB, 'alias_symbol'):
        return VOCAB.alias_symbol.get(symbol, symbol)
    return symbol


def _upper_triangle_mask(L: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones((L, L), device=device, dtype=torch.bool)
    return torch.triu(mask, diagonal=1)


def _pep_symbols(S_pep: torch.Tensor) -> List[str]:
    symbols = []
    for i in S_pep.tolist():
        symbol = VOCAB.idx_to_symbol(int(i))
        symbols.append(_resolve_symbol(symbol))
    return symbols


def _valid_mask_ht(L: int, device: torch.device, allow_alt: bool = False, min_end: int = 1) -> torch.Tensor:
    mask = torch.zeros((L, L), device=device, dtype=torch.bool)
    if L < 2:
        return mask
    if not allow_alt:
        mask[0, L - 1] = True
        return mask
    # Allow near-end alternatives if enabled.
    i_candidates = [0]
    j_candidates = list(range(max(min_end, 1), L))
    for i in i_candidates:
        for j in j_candidates:
            if i < j:
                mask[i, j] = True
    return mask


def _valid_mask_ss(symbols: List[str], min_sep: int, device: torch.device) -> torch.Tensor:
    L = len(symbols)
    mask = torch.zeros((L, L), device=device, dtype=torch.bool)
    if L < 2:
        return mask
    is_cys = [s == 'C' for s in symbols]
    for i in range(L):
        if not is_cys[i]:
            continue
        for j in range(i + min_sep, L):
            if is_cys[j]:
                mask[i, j] = True
    return mask


def _valid_mask_iso(symbols: List[str], device: torch.device) -> torch.Tensor:
    L = len(symbols)
    mask = torch.zeros((L, L), device=device, dtype=torch.bool)
    if L < 2:
        return mask
    is_lys = [s == 'K' for s in symbols]
    is_asp = [s == 'D' for s in symbols]
    for i in range(L):
        for j in range(i + 1, L):
            if (is_lys[i] and is_asp[j]) or (is_asp[i] and is_lys[j]):
                mask[i, j] = True
    return mask


class CyclizationHeads(nn.Module):
    def __init__(
        self,
        d_cyc: int,
        type_hidden: int = 128,
        pair_hidden: int = 128,
        min_sep_all: int = 1,
        min_sep_ss: int = 3,
        allow_ht_alt_pairs: bool = False,
        ht_alt_min_end: int = 1,
        mask_mode: str = 'teacher',
        relax_residue_constraints: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.min_sep_ss = min_sep_ss
        self.min_sep_all = min_sep_all
        self.allow_ht_alt_pairs = allow_ht_alt_pairs
        self.ht_alt_min_end = ht_alt_min_end
        self.mask_mode = mask_mode
        self.relax_residue_constraints = relax_residue_constraints
        self.eps = eps

        self.type_mlp = nn.Sequential(
            nn.Linear(d_cyc, type_hidden),
            nn.SiLU(),
            nn.Linear(type_hidden, len(CYCLE_TYPES)),
        )
        self.pair_mlp = nn.Sequential(
            nn.Linear(d_cyc, pair_hidden),
            nn.SiLU(),
            nn.Linear(pair_hidden, 1),
        )

    def _valid_mask_by_type(self, symbols: List[str], type_idx: int, device: torch.device) -> torch.Tensor:
        if CYCLE_TYPES[type_idx] == 'HT':
            return _valid_mask_ht(len(symbols), device, self.allow_ht_alt_pairs, self.ht_alt_min_end)
        if CYCLE_TYPES[type_idx] == 'SS':
            mask = _valid_mask_ss(symbols, self.min_sep_ss, device)
            if self.relax_residue_constraints and not mask.any():
                mask = _upper_triangle_mask(len(symbols), device)
            return mask
        if CYCLE_TYPES[type_idx] == 'ISO':
            mask = _valid_mask_iso(symbols, device)
            if self.relax_residue_constraints and not mask.any():
                mask = _upper_triangle_mask(len(symbols), device)
            return mask
        raise ValueError(f'Unknown type index {type_idx}')

    def forward_single(
        self,
        pair_cyc: torch.Tensor,
        S_pep: torch.Tensor,
        type_label: Optional[int] = None,
        type_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            pair_cyc: [L, L, d_cyc]
        Returns:
            type_logits: [3]
            pair_logits: [L, L]
            s_ij: [L, L]
        """
        device = pair_cyc.device
        L = pair_cyc.size(0)
        symbols = _pep_symbols(S_pep)

        if not bool(torch.isfinite(pair_cyc).all()):
            raise FloatingPointError('CyclizationHeads: pair_cyc input is non-finite')
        pooled = pair_cyc.mean(dim=(0, 1))
        if not bool(torch.isfinite(pooled).all()):
            raise FloatingPointError('CyclizationHeads: pooled pair_cyc is non-finite')
        if type_logits is None:
            type_logits = self.type_mlp(pooled)
        pair_logits = self.pair_mlp(pair_cyc).squeeze(-1)

        # A non-finite model output is a real failure, not a missing valid pair.
        # In particular, never turn NaN logits into an arbitrary proposed bond.
        if not bool(torch.isfinite(type_logits).all()) or not bool(torch.isfinite(pair_logits).all()):
            bad_params = [
                name for name, p in self.type_mlp.named_parameters()
                if not bool(torch.isfinite(p).all())
            ]
            max_input = float(pooled.abs().max().item()) if pooled.numel() else 0.0
            raise FloatingPointError(
                'CyclizationHeads: non-finite type logits despite finite pair features; '
                f'pooled_abs_max={max_input:.6g}, nonfinite_type_mlp_params={bad_params}'
            )

        upper_mask = _upper_triangle_mask(L, device)
        if self.mask_mode == 'teacher':
            idx = int(torch.argmax(type_logits).item()) if type_label is None else type_label
            valid_mask = self._valid_mask_by_type(symbols, idx, device)
        elif self.mask_mode == 'predicted':
            idx = int(torch.argmax(type_logits).item())
            valid_mask = self._valid_mask_by_type(symbols, idx, device)
        elif self.mask_mode == 'mixture':
            probs = torch.softmax(type_logits.float(), dim=-1)
            valid_mask = torch.zeros((L, L), device=device, dtype=torch.float32)
            for t_idx in range(len(CYCLE_TYPES)):
                mask_t = self._valid_mask_by_type(symbols, t_idx, device).float()
                valid_mask = valid_mask + probs[t_idx] * mask_t
        elif self.mask_mode in ('none', 'no_type', 'unconditional'):
            valid_mask = upper_mask
        else:
            raise ValueError(f'Unknown mask_mode {self.mask_mode}')

        sep_mask = torch.triu(torch.ones((L, L), device=device, dtype=torch.bool),
                              diagonal=max(1, self.min_sep_all))
        if valid_mask.dtype == torch.bool:
            allowed = valid_mask & upper_mask & sep_mask
            weights = allowed.float()
        else:
            weights = valid_mask.float() * (upper_mask & sep_mask).float()
            allowed = weights > 0
        if not bool(allowed.any()):
            # Strict constraints may make closure impossible (or L < 2).
            # No closure is represented by zero probability, not NaNs or an
            # chemically invalid fallback. Hard-pair selection handles this.
            return type_logits, pair_logits, torch.zeros_like(pair_logits)

        # Float32 softmax avoids fp16 overflow; invalid entries have exactly
        # zero mass even in mixture mode (no log(eps) mass on invalid bonds).
        logits = pair_logits.float() + torch.log(weights.clamp_min(self.eps))
        logits = logits.masked_fill(~allowed, float('-inf'))
        probs = torch.softmax(logits.reshape(-1), dim=0).reshape(L, L)
        s_ij = probs.to(dtype=pair_logits.dtype)
        return type_logits, pair_logits, s_ij

    def forward(
        self,
        pair_cyc: torch.Tensor,
        pep_indices: List[torch.Tensor],
        S: torch.Tensor,
        type_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns batch-level tensors padded into full [N, N] for pair outputs.
        """
        device = pair_cyc.device
        N = pair_cyc.size(0)
        bs = len(pep_indices)
        type_logits_all = []
        pair_logits_full = pair_cyc.new_zeros((N, N))
        s_ij_full = pair_cyc.new_zeros((N, N))

        for b in range(bs):
            pep_idx = pep_indices[b]
            if pep_idx.numel() == 0:
                type_logits_all.append(torch.zeros(len(CYCLE_TYPES), device=device))
                continue
            pair_cyc_b = pair_cyc[pep_idx][:, pep_idx]
            S_pep = S[pep_idx]
            type_label = int(type_labels[b].item()) if type_labels is not None else None
            type_logits, pair_logits, s_ij = self.forward_single(
                pair_cyc_b, S_pep, type_label=type_label
            )
            type_logits_all.append(type_logits)
            pair_logits_full[pep_idx[:, None], pep_idx[None, :]] = pair_logits
            s_ij_full[pep_idx[:, None], pep_idx[None, :]] = s_ij

        return {
            'type_logits': torch.stack(type_logits_all, dim=0),
            'pair_logits_full': pair_logits_full,
            's_ij_full': s_ij_full,
        }
