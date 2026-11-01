"""Global-budget, sequential GPU microbatched SMC for APCyc.

This module is only called from APCycLDMPepDesign.sample when
sample_opt.smc.enabled is true.

At a period boundary:
  * evolve each GPU microbatch through the SAME reverse-diffusion window
  * transfer normalized latent state and raw guidance scores to CPU
  * compute ONE global ESS and ONE global resampling decision
  * reassemble the next window's microbatches using globally selected parents
Only generated atoms/residues of duplicated descendants receive rejuvenation.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
import warnings

import torch

from apcyc.smc import SMCController


def _as_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, list):
        return [_as_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_as_cpu(v) for v in x)
    if isinstance(x, dict):
        return {k: _as_cpu(v) for k, v in x.items()}
    return x


def _merge_outputs(groups):
    """Preserve the original APCyc list/tuple or tensor output contract."""
    if not groups:
        return []
    if torch.is_tensor(groups[0]):
        return torch.cat(groups, dim=0)
    values = [element for group in groups for element in group]
    return tuple(values) if isinstance(groups[0], tuple) else values


def _merge_pair_outputs(groups):
    if not groups:
        return None
    result = {}
    for key in ('type_logits', 'type_hat', 'pair_hat'):
        result[key] = torch.cat([_as_cpu(group[key]) for group in groups], dim=0)
    result['pair_traj'] = None
    return result


def _segments(num_steps, period, *, force_stop=None):
    if num_steps < 1 or period < 1:
        raise ValueError('num_steps and smc.resample_every must be >= 1')
    windows = []
    top = num_steps
    if force_stop is not None and not 1 <= int(force_stop) <= num_steps:
        raise ValueError('SMC forced scoring boundary must be in [1, num_steps]')
    while top >= 1:
        bottom = max(1, top - period + 1)
        # Score exactly when cyclization guidance is first allowed, even when
        # that makes one window shorter than the configured resampling period.
        if force_stop is not None and bottom < force_stop <= top:
            bottom = int(force_stop)
        windows.append((top, bottom))
        top = bottom - 1
    return windows


class GlobalWeights:
    """One CPU log-weight vector and ESS across ALL particles of one target."""
    def __init__(self, count, *, threshold=0.5, weight_scale=5.,
                 n_observations=1, nonfinite_policy='penalize'):
        self.count = int(count)
        if self.count < 1:
            raise ValueError('SMC population must have at least one particle')
        self.threshold = float(threshold)
        self.scale = float(weight_scale)
        self.n_observations = max(1, int(n_observations))
        self.nonfinite_policy = str(nonfinite_policy)
        if not 0 <= self.threshold <= 1 or not math.isfinite(self.threshold):
            raise ValueError('smc.resample_threshold must be in [0, 1]')
        if not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError('smc.weight_scale must be finite and >= 0')
        if self.nonfinite_policy not in ('raise', 'penalize'):
            raise ValueError('smc.nonfinite_policy must be raise or penalize')
        self.log_weights = torch.full((self.count,), -math.log(self.count), dtype=torch.float64)
        # The reference potential before the first observation is zero.
        # Crucially, keep this per-particle quantity across resampling; resetting
        # only the log weights must NOT cause the next update to reapply Phi_t.
        self.previous_potential = torch.zeros(self.count, dtype=torch.float64)
        self.diagnostics = []
        self.ancestors = torch.arange(self.count, dtype=torch.long)
        self.warned_nonfinite = False

    def update(self, t, raw_scores, *, allow_resample=True):
        scores = torch.as_tensor(raw_scores, dtype=torch.float64, device='cpu').reshape(-1)
        if scores.numel() != self.count:
            raise ValueError(f'Global SMC expected {self.count} scores; got {scores.numel()}')
        valid = torch.isfinite(scores)
        # Even with finite Phi_t and Phi_prev, a difference could overflow.
        delta = scores - self.previous_potential
        valid &= torch.isfinite(delta)
        count_bad = int((~valid).sum().item())
        if count_bad:
            if self.nonfinite_policy == 'raise':
                raise ValueError(f'Global SMC t={t}: {count_bad} non-finite particle scores')
            if not self.warned_nonfinite:
                warnings.warn(f'Global SMC t={t}: {count_bad} non-finite score(s); '
                              'penalizing invalid particles', RuntimeWarning, stacklevel=2)
                self.warned_nonfinite = True
        if bool(valid.any()):
            # A constant potential contributes exactly zero after its initial
            # observation. This avoids repeatedly multiplying by exp(beta*Phi).
            # Keep the v5 effective-temperature scaling for compatibility.
            increment = torch.zeros(self.count, dtype=torch.float64)
            increment[valid] = (self.scale / self.n_observations) * delta[valid]
            if count_bad:
                # Penalize invalid observations, but do NOT install an
                # arbitrary fake potential as the previous potential. Doing
                # so would create a huge, artificial positive rebound when
                # a particle's score becomes finite again.
                increment[~valid] = increment[valid].min() - 20.0
            if not bool(torch.isfinite(increment).all()):
                raise ValueError(f'Global SMC t={t}: non-finite incremental weights')
            self.log_weights += increment
            self.log_weights -= torch.logsumexp(self.log_weights, dim=0)
            self.previous_potential[valid] = scores[valid]
        # When no scores are valid, retain the previous weights rather than
        # creating NaN or an artificial resampling decision. Do not update
        # previous potentials on such a skipped observation.
        probabilities = self.log_weights.softmax(dim=0)
        ess = float((probabilities.square().sum()).reciprocal())
        resample = bool(bool(valid.any()) and allow_resample and self.count > 1 and
                        self.threshold > 0 and ess < self.threshold * self.count)
        parents = torch.arange(self.count, dtype=torch.long)
        unique_before = int(torch.unique(self.ancestors).numel())
        if resample:
            # Systematic resampling has O(P) cost and a single random offset.
            u0 = float(torch.rand(()).item()) / self.count
            positions = u0 + torch.arange(self.count, dtype=torch.float64) / self.count
            cdf = probabilities.cumsum(0)
            cdf[-1] = 1.0
            parents = torch.searchsorted(cdf.contiguous(), positions.contiguous()).clamp(max=self.count - 1)
            self.ancestors = self.ancestors.index_select(0, parents)
            self.previous_potential = self.previous_potential.index_select(0, parents).clone()
            self.log_weights.fill_(-math.log(self.count))
        multiplicity = Counter(parents.tolist())
        duplicated = sum(max(0, c - 1) for c in multiplicity.values()) if resample else 0
        self.diagnostics.append({
            't': int(t), 'ess': ess, 'population': self.count,
            'resampled': resample, 'duplicated_descendants': duplicated,
            'unique_original_ancestors': int(torch.unique(self.ancestors).numel()),
            'unique_ancestors_before': unique_before,
            'invalid_scores': count_bad,
            'increment_min': float(increment[valid].min()) if bool(valid.any()) else 0.0,
            'increment_max': float(increment[valid].max()) if bool(valid.any()) else 0.0,
        })
        return parents, resample


def _decayed_rejuvenation_scale(base_scale, *, t, num_steps, min_t, power=1.0):
    """Anneal latent mutation to exactly zero in the last `min_t` steps.

    Reverse diffusion counts down: t=T,...,1,0. We perturb the state at the
    *next* t (bottom-1), not the score-observation timestep (bottom).
    """
    if (num_steps < 1 or t < 0 or t > num_steps or min_t < 0 or
            not math.isfinite(base_scale) or base_scale < 0 or
            not math.isfinite(power) or power <= 0):
        raise ValueError('Invalid SMC rejuvenation schedule parameters')
    if base_scale == 0 or t <= min_t or num_steps <= min_t:
        return 0.0
    progress = (t - min_t) / (num_steps - min_t)
    return float(base_scale * progress ** power)


def _jitter_duplicated(X, H, parents, mask, atom_mask, *, scale, t, min_t,
                       latent_n_channel=None):
    """Optionally rejuvenate copies only, leaving receptor/context unmodified."""
    if scale <= 0 or t <= min_t:
        return X, H
    counts = Counter()
    m = mask.to(dtype=torch.bool, device='cpu')
    # In APCyc sampling the generated atom mask is overwritten to allow the
    # autoencoder's latent channels. The original atom_mask need not include
    # all of them, so use explicit latent channel count instead.
    n_lat = X.shape[2] if latent_n_channel is None else int(latent_n_channel)
    if not 1 <= n_lat <= X.shape[2]:
        raise ValueError('Invalid latent_n_channel for SMC rejuvenation')
    channels = torch.arange(X.shape[2])[None, :, None] < n_lat
    xmask = m[:, None, None] & channels
    for child, parent in enumerate(parents.tolist()):
        counts[parent] += 1
        if counts[parent] == 1:
            continue  # Keep one exact copy of every selected parent.
        X[child] += scale * torch.randn_like(X[child]) * xmask
        H[child] += scale * torch.randn_like(H[child]) * m[:, None]
    return X, H


def _slice_complex(X, S, mask, position_ids, lengths, atom_mask, L):
    lengths = torch.as_tensor(lengths, dtype=torch.long, device=X.device).reshape(-1)
    if not bool((lengths > 0).all()) or int(lengths.sum().item()) != len(S):
        raise ValueError('Invalid APCyc flattened graph lengths')
    pos = 0
    for b, length in enumerate(lengths.tolist()):
        end = pos + length
        yield (X[pos:end], S[pos:end], mask[pos:end], position_ids[pos:end],
               atom_mask[pos:end], None if L is None else L[b:b+1])
        pos = end


def _repeat_input(group, n, *, device):
    x, s, mask, ids, atom_mask, l = group
    x = x.to(device).repeat(n, *([1] * (x.ndim - 1)))
    s = s.to(device).repeat(n)
    mask = mask.to(device).repeat(n)
    ids = ids.to(device).repeat(n)
    atom_mask = atom_mask.to(device).repeat(n, *([1] * (atom_mask.ndim - 1)))
    length = group[0].size(0)
    lengths = torch.full((n,), length, dtype=torch.long, device=device)
    if l is not None:
        l = l.to(device).repeat(n, *([1] * (l.ndim - 1)))
    return x, s, mask, ids, lengths, atom_mask, l


@torch.no_grad()
def _sample_one(model, group, sample_opt, *, particles, microbatch_size,
                return_pair=False, return_tensor=False, optimize_sidechain=True):
    if int(particles) < 1 or int(microbatch_size) < 1:
        raise ValueError('Population and GPU microbatch size must be positive')
    config = dict(sample_opt['smc'])
    if sample_opt.get('return_pair_trajectory', False):
        raise NotImplementedError('Global GPU microbatching does not store full pair trajectories; '
                                  'set return_pair_trajectory: false')
    if config.get('score', 'dist_energy') == 'property':
        raise NotImplementedError('Global microbatched SMC currently supports dist_energy and cycle scores, not property predictors')
    steps = int(model.diffusion.num_steps)
    period = int(config.get('resample_every', config.get('stride', 4)))
    score_mode = config.get('score', 'dist_energy')
    cycle_t = int(config.get('cycle_start_t', max(1, steps // 2)))
    force_cycle_boundary = (cycle_t if score_mode == 'dist_energy'
                            and float(config.get('cycle_weight', 1.0)) > 0 else None)
    windows = _segments(steps, period, force_stop=force_cycle_boundary)
    score_start = int(config.get('start_t', steps))
    score_end = int(config.get('end_t', 1))
    if score_end < 1 or score_start > steps or score_end > score_start:
        raise ValueError('Invalid smc.start_t/end_t')
    n_scores = sum(score_end <= low <= score_start for _, low in windows)
    if n_scores == 0:
        raise ValueError('SMC score window excludes all resampling boundaries')
    scorer_options = dict(config)
    scorer_options.setdefault('dist_mean', getattr(model, 'consec_dist_mean', None))
    scorer_options.setdefault('dist_std', getattr(model, 'consec_dist_std', None))
    scorer_options.update({'_score_only': True, 'resample_threshold': 0})
    global_weights = GlobalWeights(
        particles, threshold=config.get('resample_threshold', 0.5),
        weight_scale=config.get('weight_scale', 5.), n_observations=n_scores,
        nonfinite_policy=config.get('nonfinite_policy', 'penalize'))
    if score_mode == 'dist_energy' and (scorer_options['dist_mean'] is None or scorer_options['dist_std'] is None):
        raise ValueError('SMC dist_energy needs model.set_consec_dist calibration or smc.dist_mean/dist_std')
    rejuvenation = float(config.get('rejuvenation_scale', 0.01))
    rejuvenation_min_t = int(config.get('rejuvenation_min_t', 4))
    rejuvenation_decay_power = float(config.get('rejuvenation_decay_power', 1.0))
    # Validate even when no resampling occurs in a particular run.
    _decayed_rejuvenation_scale(rejuvenation, t=steps, num_steps=steps,
                               min_t=rejuvenation_min_t, power=rejuvenation_decay_power)

    state_X = state_H = None  # CPU arrays shaped [P, nodes_per_target, ...]
    x0, s0, m0, p0, am0, l0 = group
    length = int(x0.size(0))
    final_X, final_S, final_metrics, final_pair = [], [], [], []
    for top, bottom in windows:
        is_final = bottom == 1
        has_score = score_end <= bottom <= score_start
        next_x, next_h, all_scores = [], [], []
        for first in range(0, particles, microbatch_size):
            count = min(microbatch_size, particles - first)
            batch = _repeat_input(group, count, device=x0.device)
            observer_opt = {**scorer_options, 'start_t': bottom, 'end_t': bottom, 'stride': 1}
            observer = SMCController([length], count, steps, observer_opt)
            if state_X is None:
                initial = None
            else:
                initial = (state_X[first:first+count].reshape(count * length, *state_X.shape[2:]).to(x0.device),
                           state_H[first:first+count].reshape(count * length, *state_H.shape[2:]).to(x0.device))
            # Pass an internal segment descriptor. The normal model.sample path
            # remains unchanged when this descriptor is absent.
            seg_opt = dict(sample_opt)
            seg_opt.pop('smc', None)
            seg_opt['_smc_segment'] = {
                'observer': observer, 'initial_state': initial,
                'start_t': top, 'stop_t': bottom,
            }
            output = model.sample(*batch[:6], L=batch[6], sample_opt=seg_opt,
                                  return_pair=return_pair if is_final else False,
                                  return_tensor=return_tensor,
                                  optimize_sidechain=optimize_sidechain)
            if has_score:
                if not hasattr(observer, 'last_scores'):
                    raise RuntimeError(f'SMC score not observed at t={bottom}')
                all_scores.append(observer.last_scores.reshape(-1))
            if is_final:
                if return_pair:
                    xx, ss, pp, pair = output
                    final_pair.append(_as_cpu(pair))
                else:
                    xx, ss, pp = output
                final_X.append(_as_cpu(xx))
                final_S.append(_as_cpu(ss))
                final_metrics.append(_as_cpu(pp))
            else:
                sx, sh = output
                next_x.append(sx.detach().reshape(count, length, *sx.shape[1:]).cpu())
                next_h.append(sh.detach().reshape(count, length, *sh.shape[1:]).cpu())
            del output, observer, initial, batch, seg_opt
        if has_score:
            parents, resampled = global_weights.update(
                bottom, torch.cat(all_scores), allow_resample=not is_final)
        else:
            parents = torch.arange(particles)
            resampled = False
        if not is_final:
            state_X = torch.cat(next_x, dim=0)
            state_H = torch.cat(next_h, dim=0)
            if resampled:
                state_X = state_X.index_select(0, parents).clone()
                state_H = state_H.index_select(0, parents).clone()
                jitter_scale = _decayed_rejuvenation_scale(
                    rejuvenation, t=bottom - 1, num_steps=steps,
                    min_t=rejuvenation_min_t, power=rejuvenation_decay_power)
                state_X, state_H = _jitter_duplicated(
                    state_X, state_H, parents, m0.detach().cpu(), am0.detach().cpu(),
                    scale=jitter_scale, t=bottom - 1, min_t=rejuvenation_min_t,
                    latent_n_channel=model.autoencoder.latent_n_channel)
                global_weights.diagnostics[-1]['rejuvenation_scale'] = jitter_scale
        if has_score and 'rejuvenation_scale' not in global_weights.diagnostics[-1]:
            global_weights.diagnostics[-1]['rejuvenation_scale'] = 0.0
        if bool(config.get('log_stats', False)) and has_score:
            print('[GlobalSMC]', global_weights.diagnostics[-1], flush=True)
    # Expose useful diagnostics without altering original return types.
    return (_merge_outputs(final_X), _merge_outputs(final_S),
            _merge_outputs(final_metrics), _merge_pair_outputs(final_pair),
            global_weights.diagnostics)


@torch.no_grad()
def sample_population(model, X, S, mask, position_ids, lengths, atom_mask, L,
                      sample_opt, *, return_pair=False, return_tensor=False,
                      optimize_sidechain=True):
    """Return exactly P samples per input target; no more than M GPU particles.

    Multiple receptor targets are processed sequentially; global ESS is
    per-target (never mixed across distinct receptors).
    """
    smc = sample_opt['smc']
    P = int(smc['particles'])
    M = int(smc.get('gpu_microbatch_size', 2))
    if P < 1 or M < 1:
        raise ValueError('smc.particles and smc.gpu_microbatch_size must be positive')
    M = min(M, P)
    result_X, result_S, result_ppls, result_pairs = [], [], [], []
    diagnostics = []
    for group in _slice_complex(X, S, mask, position_ids, lengths, atom_mask, L):
        xx, ss, pp, pair, diag = _sample_one(
            model, group, sample_opt, particles=P, microbatch_size=M,
            return_pair=return_pair, return_tensor=return_tensor,
            optimize_sidechain=optimize_sidechain)
        result_X.append(xx)
        result_S.append(ss)
        result_ppls.append(pp)
        if return_pair:
            result_pairs.append(pair)
        diagnostics.append(diag)
    # No output-dir coupling here; v3 runners already write under .../smc/.
    # One entry per original receptor (not per GPU microbatch).
    model._last_smc_diagnostics = diagnostics
    if return_pair:
        return (_merge_outputs(result_X), _merge_outputs(result_S),
                _merge_outputs(result_ppls), _merge_pair_outputs(result_pairs))
    return (_merge_outputs(result_X), _merge_outputs(result_S),
            _merge_outputs(result_ppls))
