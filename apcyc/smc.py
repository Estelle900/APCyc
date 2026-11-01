"""Optional per-complex particle filtering in APCyc's reverse diffusion sampler."""
from __future__ import annotations
from typing import Mapping, Any, Optional
from pathlib import Path
import torch
import warnings


def smc_output_dir(base_dir, sample_opt, *, model_is_apcyc=True):
    """Keep SMC outputs separate, even for a single-sample budget.

    Baseline outputs are unchanged when SMC is disabled. The helper is pure.
    """
    if not model_is_apcyc or not isinstance(sample_opt, Mapping):
        return str(base_dir)
    setting = sample_opt.get('smc')
    if setting is None or setting is False:
        return str(base_dir)
    if setting is True:
        options = {}
    elif isinstance(setting, Mapping):
        options = setting
    else:
        raise TypeError('sample_opt.smc must be a boolean or mapping')
    if not options.get('enabled', True):
        return str(base_dir)
    particles = int(options.get('particles', 1))
    if particles < 1:
        raise ValueError('smc.particles must be >= 1')
    subdir = options.get('output_subdir', 'smc')
    if not isinstance(subdir, str) or not subdir or subdir in ('.', '..') or ("/" in subdir) or ("\\" in subdir):
        raise ValueError('smc.output_subdir must be a nonempty single directory name')
    return str(Path(base_dir) / subdir)


def smc_budget_options(sample_opt, n_samples, *, model_is_apcyc=True):
    """Build exactly `n_samples` particles per original complex, only for SMC.

    Returns None for ordinary generation. This is called ONCE by each runner;
    the model's return_all option prevents a second sampling loop.
    """
    if not model_is_apcyc or not isinstance(sample_opt, Mapping):
        return None
    setting = sample_opt.get('smc')
    if setting is None or setting is False:
        return None
    if setting is True:
        options = {}
    elif isinstance(setting, Mapping):
        options = dict(setting)
    else:
        raise TypeError('sample_opt.smc must be a boolean or mapping')
    if not options.get('enabled', True):
        return None
    n_samples = int(n_samples)
    if n_samples < 1:
        raise ValueError('n_samples / sample_opt.num_samples must be >= 1')
    # Deprecated `smc.particles` is deliberately replaced by total output budget.
    options['particles'] = n_samples
    options['return_all'] = True
    return {**sample_opt, 'smc': options}


def smc_slice_outputs(results, batch_size, n_samples, particle_index):
    """Adapter for old runner save loops: choose particle k from all B*N outputs.

    Full results are grouped by target: [target0 particle0..P-1, target1 ...].
    """
    if not 0 <= int(particle_index) < int(n_samples):
        raise IndexError('Invalid SMC particle index')
    indices = torch.arange(int(batch_size), dtype=torch.long) * int(n_samples) + int(particle_index)
    X, S, pmetric, *rest = results
    expected = int(batch_size) * int(n_samples)
    if len(X) != expected or len(S) != expected or len(pmetric) != expected:
        raise ValueError(f'SMC expected exactly {expected} outputs, got {len(X)}, {len(S)}, {len(pmetric)}')
    out = [select_samples(v, indices) for v in (X, S, pmetric)]
    if rest:
        pair = rest[0]
        if pair is None:
            out.append(None)
        else:
            sub = dict(pair)
            for name in ('type_logits', 'type_hat', 'pair_hat'):
                if name in sub and sub[name] is not None:
                    sub[name] = select_samples(sub[name], indices)
            # The full trajectory is not part of the saved per-particle record.
            sub.pop('pair_traj', None)
            out.append(sub)
    return tuple(out)


def expand_graph_batch(X, S, mask, position_ids, lengths, atom_mask, L, particles):
    """Repeat whole complexes rather than interleaving individual residues."""
    particles = int(particles)
    lengths = torch.as_tensor(lengths, device=X.device, dtype=torch.long).reshape(-1)
    if particles < 1 or not bool(torch.all(lengths > 0)) or lengths.sum().item() != X.size(0):
        raise ValueError('Invalid particle count or flattened batch lengths')
    if any(v.size(0) != X.size(0) for v in (S, mask, position_ids, atom_mask)):
        raise ValueError('Invalid flattened batch tensor sizes')
    ends = lengths.cumsum(0).tolist()
    starts = [0] + ends[:-1]
    indices = torch.cat([torch.arange(a, b, device=X.device).repeat(particles)
                         for a, b in zip(starts, ends)])
    if L is not None:
        if L.size(0) != len(lengths):
            raise ValueError('L must have one covariance matrix per complex')
        L = L.repeat_interleave(particles, 0)
    return (X.index_select(0, indices), S.index_select(0, indices),
            mask.index_select(0, indices), position_ids.index_select(0, indices),
            lengths.repeat_interleave(particles), atom_mask.index_select(0, indices),
            L, lengths)


def select_samples(values, indices):
    if torch.is_tensor(values):
        return values.index_select(0, indices.to(values.device))
    selected = [values[int(i)] for i in indices.tolist()]
    return tuple(selected) if isinstance(values, tuple) else selected


class SMCController:
    """Receptor-local particle filtering with dist_energy + late cycle guidance.

    The potential is a heuristic score, NOT an exact diffusion likelihood ratio.
    """
    def __init__(self, original_lengths, particles, num_steps, options: Optional[Mapping[str, Any]]=None):
        options = dict(options or {})
        self.lengths = torch.as_tensor(original_lengths, dtype=torch.long).reshape(-1)
        self.batch_size = self.lengths.numel()
        self.particles = int(particles)
        self.num_steps = int(num_steps)
        self.score_mode = options.get('score', 'dist_energy')
        self.start_t = int(options.get('start_t', num_steps))
        self.end_t = int(options.get('end_t', 1))
        self.stride = int(options.get('stride', 1))
        self.threshold = float(options.get('resample_threshold', 0.5))
        self.scale = float(options.get('weight_scale', 5.0))
        self.selection = options.get('selection', 'sample')
        self.property_weights = options.get('property_weights')
        self.dist_mean = options.get('dist_mean')
        self.dist_std = options.get('dist_std')
        self.dist_tolerance = float(options.get('dist_tolerance', 3))
        self.cycle_start_t = int(options.get('cycle_start_t', max(1, num_steps // 2)))
        self.cycle_weight = float(options.get('cycle_weight', 1.0))
        self.return_all = bool(options.get('return_all', False))
        self.score_only = bool(options.get('_score_only', False))
        self.nonfinite_policy = options.get('nonfinite_policy', 'penalize')
        self.pair_fallback = options.get('pair_fallback', 'unmasked')
        if self.nonfinite_policy not in ('penalize', 'raise'):
            raise ValueError('smc.nonfinite_policy must be penalize or raise')
        if self.pair_fallback not in ('unmasked', 'raise'):
            raise ValueError('smc.pair_fallback must be unmasked or raise')
        self.invalid_score_history = []
        self.repaired_pair_history = []
        self._warned_invalid_score = False
        self._warned_repair = False
        # The global microbatch orchestrator can use a singleton scoring window;
        # SMC resampling itself still requires a global population of >= 2.
        if self.particles < 2 and not options.get('_score_only', False):
            raise ValueError('SMCController needs at least 2 particles')
        if self.score_mode not in ('dist_energy', 'cycle', 'property'):
            raise ValueError('smc.score must be dist_energy, cycle, or property')
        if self.score_mode == 'dist_energy':
            if self.dist_mean is None or self.dist_std is None:
                raise ValueError(
                    'smc.score=dist_energy needs consecutive-distance mean/std in the checkpoint '
                    '(model.set_consec_dist) or smc.dist_mean and smc.dist_std in YAML')
            if not (float(self.dist_mean) > 0 and float(self.dist_std) >= 0):
                raise ValueError('smc.dist_mean must be >0 and smc.dist_std must be >=0')
        if not (1 <= self.cycle_start_t <= self.num_steps):
            raise ValueError('smc.cycle_start_t must be between 1 and num_steps')
        if not torch.isfinite(torch.tensor([self.cycle_weight, self.dist_tolerance])).all():
            raise ValueError('smc.cycle_weight and dist_tolerance must be finite')
        if self.dist_tolerance < 0 or self.cycle_weight < 0:
            raise ValueError('smc.cycle_weight and dist_tolerance must be >= 0')
        if not (1 <= self.end_t <= self.start_t <= self.num_steps) or self.stride < 1:
            raise ValueError('smc expects 1 <= end_t <= start_t <= num_steps, stride >= 1')
        if not 0 <= self.threshold <= 1:
            raise ValueError('smc.resample_threshold must be in [0, 1]')
        if not torch.isfinite(torch.tensor(self.scale)) or self.scale < 0:
            raise ValueError('smc.weight_scale must be finite and >= 0')
        if self.selection not in ('max', 'sample'):
            raise ValueError('smc.selection must be max or sample')
        self.n_observations = len(range(self.start_t, self.end_t-1, -self.stride))
        self.log_weights = None
        self.ess_history = []
        self.resampling_steps = []

    def should_score(self, t):
        return self.end_t <= t <= self.start_t and (self.start_t-t) % self.stride == 0

    def _cycle_score(self, pair_out):
        type_score = pair_out['type_logits'].log_softmax(-1).amax(-1)
        probs = pair_out['s_ij_full']
        pair_scores = []
        for idx in pair_out['pep_indices']:
            pair_scores.append(probs[idx][:, idx].amax().clamp_min(1e-12).log()
                               if idx.numel() else probs.new_tensor(-30.0))
        return type_score + torch.stack(pair_scores)

    @torch.no_grad()
    def _dist_energy_score(self, X_unorm, mask_generate):
        """Negative *existing APCyc dist_energy*, independently per particle.

        APCyc's dist_energy sums over the batch, so calling it on the entire
        duplicated batch would produce ONE score, not a particle-wise score.
        """
        from models.LDM.energies.dist import dist_energy
        scores = []
        offset = 0
        for original_len in self.lengths.tolist():
            for _ in range(self.particles):
                stop = offset + original_len
                X = X_unorm[offset:stop]
                mask = mask_generate[offset:stop]
                ids = torch.zeros(original_len, dtype=torch.long, device=X.device)
                energy = dist_energy(
                    X=X, mask_generate=mask, batch_ids=ids,
                    mean=float(self.dist_mean), std=float(self.dist_std),
                    tolerance=self.dist_tolerance)
                scores.append(-energy)
                offset = stop
        if offset != X_unorm.shape[0]:
            raise ValueError('SMC dist_energy input does not match particle-expanded lengths')
        return torch.stack(scores)

    def _property_score(self, predictor, *, H_t, X_t, mask_generate, atom_mask,
                        position_ids, ctx_edges, inter_edges, ctx_edge_attr, inter_edge_attr,
                        edge_gate, edge_bias, edge_type_ids, pair_out, p_c,
                        t_batch, batch_ids, atom_embeddings):
        if predictor is None:
            raise ValueError('smc.score=property requires property_ckpt(s)')
        y = predictor(H_t, X_t, mask_generate, atom_mask, position_ids,
                      ctx_edges, inter_edges,
                      edge_attr=torch.cat([ctx_edge_attr, inter_edge_attr]),
                      edge_gate=edge_gate, edge_bias=edge_bias, edge_type=edge_type_ids,
                      pair_struct=pair_out['pair_struct'], pair_cyc=pair_out['pair_cyc'],
                      type_probs=p_c, s_ij=pair_out['s_ij_full'], t=t_batch,
                      batch_ids=batch_ids, atom_embeddings=atom_embeddings)
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or y.size(0) != self.batch_size * self.particles:
            raise ValueError('Property predictor must output [batch*particles, n_properties]')
        w = y.new_ones(y.size(1)) if self.property_weights is None else torch.as_tensor(
            self.property_weights, device=y.device, dtype=y.dtype)
        if w.ndim != 1 or w.numel() != y.size(1):
            raise ValueError('smc.property_weights must match predictor output dimension')
        return (y*w).sum(-1)

    @torch.no_grad()
    def repair_pair_probabilities(self, t, pair_out):
        """Repair only the all-masked cyclization softmax in SMC mode.

        APCyc's head can softmax an entirely -inf masked pair matrix when its
        residue/type constraints leave no allowed bond. This yields NaNs even
        though the original unmasked pair logits are finite. For this *specific*
        failure, fall back to a distribution over upper-triangular peptide pairs.
        Non-finite raw model logits are NOT repaired: they indicate a different,
        potentially unstable diffusion/model failure.
        """
        probs = pair_out['s_ij_full']
        if bool(torch.isfinite(probs).all()):
            return
        if not bool(torch.isfinite(pair_out['type_logits']).all()):
            raise ValueError(
                f'SMC t={t}: non-finite cyclization type logits; model state may be unstable. '
                'Try disabling SMC to check baseline stability.')
        raw = pair_out.get('pair_logits_full')
        if raw is None:
            raise ValueError(f'SMC t={t}: cannot repair pair probabilities without pair_logits_full')
        fixed = probs.clone()
        repaired = []
        for particle, idx in enumerate(pair_out['pep_indices']):
            if idx.numel() == 0:
                continue
            local = fixed[idx[:, None], idx[None, :]]
            if bool(torch.isfinite(local).all()):
                continue
            if self.pair_fallback == 'raise':
                raise ValueError(
                    f'SMC t={t}, particle={particle}: invalid cyclization pair probabilities. '
                    'Set smc.pair_fallback=unmasked to allow the fallback.')
            n = idx.numel()
            if n < 2:
                raise ValueError(
                    f'SMC t={t}, particle={particle}: peptide has fewer than two residues; '
                    'cannot form a cyclization pair.')
            upper = torch.ones((n, n), dtype=torch.bool, device=idx.device).triu_(diagonal=1)
            pair_logits = raw[idx[:, None], idx[None, :]][upper]
            if not bool(torch.isfinite(pair_logits).all()):
                raise ValueError(
                    f'SMC t={t}, particle={particle}: non-finite unmasked pair logits. '
                    'This indicates a model/diffusion numerical problem, not just an empty pair mask.')
            fallback = torch.zeros_like(local)
            fallback[upper] = torch.softmax(pair_logits.float(), dim=0).to(fallback.dtype)
            fixed[idx[:, None], idx[None, :]] = fallback
            repaired.append(particle)
        if not bool(torch.isfinite(fixed).all()):
            raise ValueError(
                f'SMC t={t}: non-finite probabilities remain outside peptide blocks. '
                'Check the diffusion state and model outputs.')
        pair_out['s_ij_full'] = fixed
        if repaired:
            self.repaired_pair_history.append((int(t), repaired))
            if not self._warned_repair:
                warnings.warn(
                    f'SMC t={t}: repaired invalid cyclization probabilities for '
                    f'{len(repaired)} particle(s) using unmasked pair logits. '
                    'This relaxes pair-type constraints; check generated bond validity. '
                    'To reject such cases, use smc.pair_fallback=raise.',
                    RuntimeWarning, stacklevel=2)
                self._warned_repair = True

    @torch.no_grad()
    def observe(self, t, *, pair_out, predictor=None, **inputs):
        if not self.should_score(t):
            return
        if self.score_mode == 'dist_energy':
            if 'X_unorm' not in inputs:
                raise ValueError('SMC dist_energy requires unnormalized X_unorm')
            score = self._dist_energy_score(inputs['X_unorm'], inputs['mask_generate'])
            if t <= self.cycle_start_t and self.cycle_weight > 0:
                score = score + self.cycle_weight * self._cycle_score(pair_out)
        elif self.score_mode == 'cycle':
            score = self._cycle_score(pair_out)
        else:
            # X_unorm is used only by the default distance-energy score.
            score = self._property_score(predictor, pair_out=pair_out,
                                         **{k: v for k, v in inputs.items() if k != 'X_unorm'})
        if score.numel() != self.batch_size * self.particles:
            raise ValueError('SMC requires one score per particle')
        score = score.float().reshape(self.batch_size, self.particles)
        # Raw scores are gathered from GPU microbatches before ONE global ESS update.
        self.last_scores = score.detach().cpu().clone()
        # No local ESS update; all raw scores are combined globally by smc_global.
        if self.score_only:
            return
        valid = torch.isfinite(score)
        if not bool(valid.all()):
            bad = (~valid).nonzero(as_tuple=False)
            groups = torch.unique(bad[:, 0]).tolist()
            self.invalid_score_history.append((int(t), bad.detach().cpu().tolist()))
            if self.nonfinite_policy == 'raise':
                raise ValueError(
                    f'SMC t={t}: {len(bad)} non-finite particle score(s) in '
                    f'complex(es) {groups}. Check type_logits/s_ij_full/predictor outputs.')
            if not self._warned_invalid_score:
                warnings.warn(
                    f'SMC t={t}: {len(bad)} non-finite particle score(s) '
                    f'in complex(es) {groups}; penalizing invalid particles '
                    'and skipping scoring for groups with no valid particles. '
                    'To raise instead, set smc.nonfinite_policy=raise.',
                    RuntimeWarning, stacklevel=2)
                self._warned_invalid_score = True
        any_valid = valid.any(dim=-1, keepdim=True)
        # Each receptor is independent: never normalize scores across receptors.
        # A fully invalid receptor retains its previous particle weights.
        least_finite = score.masked_fill(~valid, float('inf')).amin(dim=-1, keepdim=True)
        least_finite = torch.where(any_valid, least_finite, torch.zeros_like(least_finite))
        safe_score = torch.where(valid, score, least_finite - 20.0)
        safe_score = safe_score - safe_score.amax(dim=-1, keepdim=True)
        increments = (self.scale / self.n_observations) * safe_score
        increments = torch.where(any_valid, increments, torch.zeros_like(increments))
        if self.log_weights is None:
            self.log_weights = torch.zeros_like(increments)
        self.log_weights += increments
        self.log_weights -= self.log_weights.logsumexp(-1, keepdim=True)

    def normalized_weights(self, device=None):
        if self.log_weights is None:
            w = torch.full((self.batch_size, self.particles), 1/self.particles)
            return w.to(device) if device is not None else w
        return self.log_weights.softmax(-1)

    @staticmethod
    def systematic_indices(weights):
        n = weights.size(1)
        starts = torch.rand(weights.size(0), 1, device=weights.device, dtype=weights.dtype) / n
        positions = starts + torch.arange(n, device=weights.device, dtype=weights.dtype)[None] / n
        boundaries = weights.cumsum(-1).contiguous()
        boundaries[:, -1] = 1.0
        return torch.searchsorted(boundaries, positions.contiguous()).clamp(max=n-1)

    @torch.no_grad()
    def resample_step(self, t, X_next, H_next):
        if t <= 1 or not self.should_score(t) or self.threshold == 0:
            return X_next, H_next
        weights = self.normalized_weights(X_next.device)
        ess = weights.square().sum(-1).reciprocal()
        self.ess_history.append((int(t), ess.detach().cpu().tolist()))
        change = ess < self.threshold * self.particles
        if not bool(change.any()):
            return X_next, H_next
        parents = self.systematic_indices(weights)
        parents[~change] = torch.arange(self.particles, device=parents.device)
        X_new, H_new, offset = [], [], 0
        for b, length in enumerate(self.lengths.tolist()):
            count = length * self.particles
            x = X_next[offset:offset+count].reshape(self.particles, length, *X_next.shape[1:])
            h = H_next[offset:offset+count].reshape(self.particles, length, *H_next.shape[1:])
            X_new.append(x.index_select(0, parents[b]).reshape(count, *X_next.shape[1:]))
            H_new.append(h.index_select(0, parents[b]).reshape(count, *H_next.shape[1:]))
            offset += count
        self.log_weights[change] = -torch.log(torch.tensor(float(self.particles), device=weights.device))
        self.resampling_steps.append((int(t), change.detach().cpu().tolist()))
        return torch.cat(X_new), torch.cat(H_new)

    def final_indices(self):
        weights = self.normalized_weights()
        chosen = weights.argmax(-1) if self.selection == 'max' else torch.multinomial(weights, 1).squeeze(-1)
        return chosen + self.particles * torch.arange(self.batch_size, device=chosen.device)
