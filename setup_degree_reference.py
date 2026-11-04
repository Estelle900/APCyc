#!/usr/bin/env python3
"""Initialize and cache degree reference from setup_latent_guidance.yaml.

Run from APCyc repo root. Uses node-weighted P90 of valid EGNN degrees.
Does not run the autoencoder or allocate graph edges.
Can optionally embed the cached reference in a new whole-model checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
import torch
import yaml


SCHEMA = 2
REFERENCE_METHOD = "node_weighted_p90"
PERCENTILE = 0.90


def dataset_fingerprint(config: dict) -> str:
    """Hash dataset specification AND index bytes to invalidate stale cache."""
    cfg = json.dumps(config['dataset'], sort_keys=True, default=str).encode('utf-8')
    h = hashlib.sha256(cfg)
    for split_name, spec in sorted(config['dataset'].items()):
        for item in (spec if isinstance(spec, list) else [spec]):
            index = item.get('specify_index') if isinstance(item, dict) else None
            if index:
                index_path = Path(index)
                if not index_path.is_file():
                    raise FileNotFoundError(f'Missing dataset index: {index_path}')
                h.update(split_name.encode('utf-8'))
                h.update(index_path.read_bytes())
    return h.hexdigest()


def accumulate_batch_degrees(batch: dict, degree_hist: dict[int, int] | None = None) -> tuple[int, int, int, int, int]:
    """Count valid graph nodes; APCyc's variadic_meshgrid includes all self-edges.

    Returns (degree_sum, valid_node_count, graph_count, min_degree, max_degree).
    If supplied, degree_hist[n] accumulates the number of valid nodes with degree n.
    Graph-level degree equals # valid nodes, for each valid node.
    Generated residues are treated as active latent nodes, as in diffusion.forward.
    """
    lengths = torch.as_tensor(batch['lengths'], dtype=torch.long).flatten().tolist()
    atom_mask = torch.as_tensor(batch['atom_mask']).bool()
    mask_generate = torch.as_tensor(batch['mask']).bool().view(-1)
    if atom_mask.ndim != 2:
        raise ValueError(f'Expected [N, channel] atom_mask, got {atom_mask.shape}')
    if atom_mask.shape[0] != sum(lengths) or mask_generate.numel() != sum(lengths):
        raise ValueError('Batch lengths do not match mask/atom_mask shape')
    # Diffusion replaces generated residue channels by active latent channels.
    present = atom_mask.any(dim=-1) | mask_generate
    offset = total_degree = valid_nodes = graphs = 0
    min_deg, max_deg = None, 0
    for length in lengths:
        if length <= 0:
            raise ValueError('Found non-positive example length')
        n = int(present[offset:offset+length].sum().item())
        offset += length
        if n <= 0:
            continue
        # n valid residues each have n incoming messages, including self.
        total_degree += n * n
        if degree_hist is not None:
            degree_hist[n] = degree_hist.get(n, 0) + n
        valid_nodes += n
        graphs += 1
        min_deg = n if min_deg is None else min(min_deg, n)
        max_deg = max(max_deg, n)
    return total_degree, valid_nodes, graphs, min_deg or 0, max_deg


def node_weighted_percentile(degree_hist: dict[int, int], percentile: float = PERCENTILE) -> float:
    """Smallest degree with cumulative node weight >= specified percentile.

    A graph with n valid nodes contributes n observations of degree n.
    This is a node-weighted *empirical* percentile, without interpolation.
    """
    if not 0 < percentile <= 1:
        raise ValueError(f'Percentile must be in (0, 1], got {percentile}')
    if not degree_hist or any(d <= 0 or count <= 0 for d, count in degree_hist.items()):
        raise ValueError('Degree histogram must contain positive degrees and weights')
    n_nodes = sum(degree_hist.values())
    rank = math.ceil(percentile * n_nodes)
    cumulative = 0
    for degree, weight in sorted(degree_hist.items()):
        cumulative += weight
        if cumulative >= rank:
            return float(degree)
    raise AssertionError('Degree histogram percentile calculation failed')


def compute_reference(config: dict, fingerprint: str) -> dict:
    # Keep imports lazy so tests and cache reads do not require APCyc datasets.
    import models  # noqa: F401; registers model/dataset classes in APCyc
    from data import create_dataset, create_dataloader
    from tqdm.auto import tqdm

    # Follow setup_latent_guidance.py verbatim: take third return value.
    # NOTE yaml.dataset.test points at train_index.txt, so this is train data.
    _, _, dataset = create_dataset(config['dataset'])
    if dataset is None:
        raise ValueError('The YAML must supply dataset.test (as setup_latent_guidance does).')
    loader = create_dataloader(dataset, config['dataloader'])

    sum_degrees = n_nodes = n_graphs = 0
    degree_hist = {}
    degree_min, degree_max = None, 0
    for batch in tqdm(loader, desc='Caching valid EGNN degrees'):
        batch_sum, nodes, graphs, min_deg, max_deg = accumulate_batch_degrees(
            batch, degree_hist=degree_hist
        )
        sum_degrees += batch_sum
        n_nodes += nodes
        n_graphs += graphs
        if graphs:
            degree_min = min_deg if degree_min is None else min(degree_min, min_deg)
            degree_max = max(degree_max, max_deg)
    if n_nodes == 0:
        raise ValueError('Found no active residues in the selected training data.')
    # P90 across valid nodes (not across graphs): a graph with n nodes has
    # n incoming-degree observations of value n. Retain the old mean for audit.
    if sum(degree_hist.values()) != n_nodes:
        raise AssertionError('Degree histogram and valid-node count disagree')
    d_ref = node_weighted_percentile(degree_hist, PERCENTILE)
    if not math.isfinite(d_ref) or d_ref <= 0:
        raise ValueError(f'Invalid reference degree: {d_ref}')
    return {
        'schema': SCHEMA,
        'fingerprint': fingerprint,
        'd_ref': float(d_ref),
        'method': REFERENCE_METHOD,
        'percentile': 90,
        'node_weighted_mean_degree': float(sum_degrees / n_nodes),
        'total_valid_residues': int(n_nodes),
        'total_incoming_valid_edges': int(sum_degrees),
        'graphs': int(n_graphs),
        'min_graph_degree': int(degree_min),
        'max_graph_degree': int(degree_max),
        'description': 'Node-weighted 90th percentile of valid incoming degree, including self-edges, APCyc diffusion all-pairs graph',
        'split': 'dataset.test (configured to train_index.txt in setup_latent_guidance.yaml)',
    }


def _read_cache(cache_path: Path):
    """Validate an existing cache; never trust an invalid/missing d_ref."""
    if not cache_path.is_file():
        return None
    try:
        cached = json.loads(cache_path.read_text(encoding='utf-8'))
        d_ref = float(cached['d_ref'])
        if (cached.get('schema') != SCHEMA
                or cached.get('method') != REFERENCE_METHOD
                or cached.get('percentile') != 90
                or not math.isfinite(d_ref) or d_ref <= 0):
            raise ValueError('Cache is not node-weighted P90 schema v2, or d_ref is invalid')
        cached['d_ref'] = d_ref
        return cached
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f'[degree-ref] Invalid cache at {cache_path}: {exc}; will recompute if possible.')
        return None


def _load_config_and_fingerprint(config_path: Path):
    """Load fingerprint if training data is accessible, otherwise return None.

    Generation on another machine can still use a previously computed cache
    without requiring the original training dataset. If no cache exists,
    missing config/index is a hard error rather than silently reverting to sum.
    """
    try:
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        fingerprint = dataset_fingerprint(config)
    except FileNotFoundError:
        return None, None
    return config, fingerprint


@contextmanager
def _cache_lock(cache_path: Path):
    """Serialize first-time dataset scans across training ranks/processes."""
    import fcntl   # APCyc generation and training target Linux.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(cache_path) + '.lock')
    with lock_path.open('a+b') as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_or_compute(config_path: Path, cache_path: Path, force: bool = False) -> dict:
    """Prefer cache, otherwise scan setup_latent_guidance data and cache d_ref.

    If the training index is locally available, compare its fingerprint to
    detect stale caches. If absent, a valid cache can still be used in generation.
    """
    config_path = Path(config_path)
    cache_path = Path(cache_path)
    if not force:
        cached = _read_cache(cache_path)
        if cached is not None:
            config, fingerprint = _load_config_and_fingerprint(config_path)
            if fingerprint is None or cached.get('fingerprint') == fingerprint:
                if fingerprint is None:
                    print('[degree-ref] Training index unavailable; using existing cache without fingerprint verification.')
                print(f'[degree-ref] Loaded cached d_ref={cached["d_ref"]:.5g} from {cache_path}')
                return cached
            print('[degree-ref] Dataset fingerprint changed; recomputing reference.')

    # Multiple training ranks may reach this path at the same time.
    with _cache_lock(cache_path):
        config, fingerprint = _load_config_and_fingerprint(config_path)
        if not force:
            cached = _read_cache(cache_path)  # another rank may have populated it
            if cached is not None and (fingerprint is None or cached.get('fingerprint') == fingerprint):
                print(f'[degree-ref] Loaded cached d_ref={cached["d_ref"]:.5g} from {cache_path}')
                return cached
        if config is None:
            raise FileNotFoundError(
                'Cannot initialize d_ref: no valid cache at '
                f'{cache_path}, and training config/index unavailable at {config_path}. '
                'Run once on a machine with the training dataset to populate the cache.'
            )
        result = compute_reference(config, fingerprint)
        # Use a unique temporary file and atomic replacement.
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', prefix=cache_path.name + '.',
            suffix='.tmp', dir=cache_path.parent, delete=False,
        ) as f:
            tmp = Path(f.name)
            json.dump(result, f, indent=2, sort_keys=True)
            f.write('\n')
        try:
            os.replace(tmp, cache_path)
        finally:
            tmp.unlink(missing_ok=True)
        print(f'[degree-ref] Scanned {result["graphs"]} graphs; d_ref={result["d_ref"]:.5g}; cached at {cache_path}')
        return result


def configure_diffusion_model(model, d_ref: float, *, no_upscale: bool = False) -> None:
    """Enable only the diffusion AMEGNN; leave autoencoder sum aggregation as is.

    The reference is cached in a non-persistent buffer: torch.save(model)
    preserves it; state_dict intentionally stays compatible with old weights.
    """
    if not math.isfinite(d_ref) or d_ref <= 0:
        raise ValueError('d_ref must be finite and positive')
    try:
        encoder = model.diffusion.eps_net.encoder
    except AttributeError as exc:
        raise TypeError('Expected APCyc model.diffusion.eps_net.encoder') from exc
    layers = [encoder._modules[f'gcl_{i}'] for i in range(encoder.n_layers)]
    layers.append(encoder.out_layer)
    for layer in layers:
        # PyTorch whole-model checkpoints may have been created before any
        # additional attributes existed, so add lazily on loaded objects.
        old = getattr(layer, '_apcyc_degree_reference', None)
        device = next(layer.parameters()).device
        ref = torch.tensor(float(d_ref), device=device, dtype=torch.float32)
        if old is None:
            layer.register_buffer('_apcyc_degree_reference', ref, persistent=False)
        else:
            layer._buffers['_apcyc_degree_reference'] = ref
        layer._apcyc_degree_no_upscale = bool(no_upscale)
    encoder._apcyc_degree_reference_value = float(d_ref)
    encoder._apcyc_degree_reference_method = REFERENCE_METHOD
    encoder._apcyc_degree_no_upscale = bool(no_upscale)
    print(f'[degree-ref] Enabled {len(layers)} diffusion GCL layers: d_ref={d_ref:.5g}, no_upscale={no_upscale}.')


def disable_diffusion_degree_reference(model):
    """Restore exact summed-message aggregation, including loaded checkpoints."""
    try:
        encoder = model.diffusion.eps_net.encoder
    except AttributeError as exc:
        raise TypeError('Expected APCyc model.diffusion.eps_net.encoder') from exc
    layers = [encoder._modules[f'gcl_{i}'] for i in range(encoder.n_layers)]
    layers.append(encoder.out_layer)
    for layer in layers:
        # A registered buffer must be removed from its registry, not merely set
        # to None: the source code checks for None to use original sum aggregation.
        if '_apcyc_degree_reference' in layer._buffers:
            del layer._buffers['_apcyc_degree_reference']
        elif hasattr(layer, '_apcyc_degree_reference'):
            delattr(layer, '_apcyc_degree_reference')
        layer._apcyc_degree_no_upscale = False
    if hasattr(encoder, '_apcyc_degree_reference_value'):
        delattr(encoder, '_apcyc_degree_reference_value')
    if hasattr(encoder, '_apcyc_degree_reference_method'):
        delattr(encoder, '_apcyc_degree_reference_method')
    encoder._apcyc_degree_no_upscale = False
    print('[degree-ref] Disabled; diffusion H aggregation uses original sum.')


def configure_from_cache(model, cache_path: str | Path, no_upscale: bool = False):
    """Configure model from a validated cache; do not scan the dataset."""
    record = _read_cache(Path(cache_path))
    if record is None:
        raise FileNotFoundError(f'Missing or invalid degree reference cache: {cache_path}')
    configure_diffusion_model(model, record['d_ref'], no_upscale=no_upscale)
    return record


def ensure_degree_reference(
    model,
    *,
    config_path: str | Path = 'configs/pepbench/ldm/setup_latent_guidance.yaml',
    cache_path: str | Path = 'cache/apcyc_degree_reference.json',
    no_upscale: bool = False,
    force: bool = False,
):
    """Use cached d_ref or compute it once, then attach to diffusion EGNN.

    Invoked automatically by both generate.py and train.py for APCyc models.
    The cached d_ref is authoritative when available. A checkpoint can be used
    as fallback only if it is tagged with the current P90 reference method.
    """
    try:
        encoder = model.diffusion.eps_net.encoder
    except AttributeError as exc:
        raise TypeError('Expected APCyc model.diffusion.eps_net.encoder') from exc
    layers = [encoder._modules[f'gcl_{i}'] for i in range(encoder.n_layers)]
    layers.append(encoder.out_layer)

    previous = getattr(layers[0], '_apcyc_degree_reference', None)
    try:
        record = get_or_compute(Path(config_path), Path(cache_path), force=force)
    except FileNotFoundError:
        # Whole-model checkpoints may already carry a non-persistent buffer.
        if (not force and previous is not None
                and getattr(encoder, '_apcyc_degree_reference_method', None) == REFERENCE_METHOD):
            d_ref = float(previous.item())
            if math.isfinite(d_ref) and d_ref > 0 and all(
                getattr(layer, '_apcyc_degree_reference', None) is not None
                for layer in layers
            ):
                print(f'[degree-ref] No cache/data; using checkpoint-embedded d_ref={d_ref:.5g}.')
                configure_diffusion_model(model, d_ref, no_upscale=no_upscale)
                return {'schema': SCHEMA, 'd_ref': d_ref, 'method': REFERENCE_METHOD, 'source': 'checkpoint'}
        raise

    d_ref = record['d_ref']
    if previous is not None and float(previous.item()) != float(d_ref):
        print(f'[degree-ref] Replacing checkpoint d_ref={float(previous.item()):.5g} '
              f'with cache/data d_ref={d_ref:.5g}.')
    configure_diffusion_model(model, d_ref, no_upscale=no_upscale)
    return record


def main():
    p = argparse.ArgumentParser(description='Cache diffusion EGNN reference degree from the dataset in setup_latent_guidance.yaml')
    p.add_argument('--config', default='configs/pepbench/ldm/setup_latent_guidance.yaml')
    p.add_argument('--cache', default='cache/apcyc_degree_reference.json')
    p.add_argument('--force', action='store_true', help='Rescan dataset even if same fingerprint is cached')
    p.add_argument('--ckpt', help='Optional input APCyc whole-model checkpoint')
    p.add_argument('--out', help='New APCyc checkpoint path (never defaults to overwriting original)')
    p.add_argument('--no-upscale', action='store_true', help='Never amplify sums for below-reference-degree nodes')
    a = p.parse_args()
    record = get_or_compute(Path(a.config), Path(a.cache), a.force)
    if a.ckpt:
        from generate import get_best_ckpt  # Same resolver as setup_latent_guidance.py
        ckpt = Path(a.ckpt if a.ckpt.endswith('.ckpt') else get_best_ckpt(a.ckpt)).resolve()
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        out = Path(a.out).resolve() if a.out else ckpt.with_name(ckpt.stem + '.degree_scaled' + ckpt.suffix)
        if out == ckpt:
            raise ValueError('Refusing to overwrite input checkpoint; use a distinct --out')
        if out.exists():
            raise FileExistsError(f'Refusing to overwrite existing output: {out}')
        model = torch.load(ckpt, map_location='cpu', weights_only=False)
        configure_diffusion_model(model, record['d_ref'], no_upscale=a.no_upscale)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, out)
        print(f'[degree-ref] Wrote new whole-model checkpoint: {out}')
    else:
        print('[degree-ref] Cache ready. Normal generate.py and train.py runs initialize automatically.')


if __name__ == '__main__':
    main()
