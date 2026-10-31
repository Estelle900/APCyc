#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats.contingency import association
from scipy.spatial.distance import pdist

from evaluation.seq_metric import align_sequences


def seq_diversity(seqs: List[str], th: float = 0.4):
    n = len(seqs)

    if n == 0:
        return 0.0, np.array([], dtype=int)

    if n == 1:
        return 1.0, np.array([1], dtype=int)

    # Condensed distance matrix
    dists = np.empty(n * (n - 1) // 2, dtype=np.float64)

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seqs[i] == seqs[j]:
                dist = 0.0
            else:
                _, sim = align_sequences(seqs[i], seqs[j])
                dist = 1.0 - sim

            dists[k] = dist
            k += 1

    if not np.isfinite(dists).all():
        raise ValueError("Invalid sequence distances")

    Z = linkage(dists, method="single")
    clusters = fcluster(Z, t=th, criterion="distance")

    diversity = len(np.unique(clusters)) / n

    return diversity, clusters


def struct_diversity(structs: np.ndarray, th: float=4.0) -> float:
    '''
    Cluster structures using CA-coordinate RMSD.

    Invalid structures are excluded from clustering.
    Their cluster labels are set to -1.

    structs: N*L*3, alpha carbon coordinates
    th: threshold for clustering (distance < th)
    '''
    structs = np.asarray(structs, dtype=np.float64)

    if structs.ndim != 3 or structs.shape[-1] != 3:
        raise ValueError("structs must have shape (N, L, 3)")

    n, L, _ = structs.shape
    clusters = np.full(n, -1, dtype=int)

    if n == 0:
        return 0.0, clusters

    if L == 0:
        return float("nan"), clusters

    # Filter structures containing NaN or Inf
    valid = np.isfinite(structs).all(axis=(1, 2))
    n_valid = int(valid.sum())

    if n_valid == 0:
        return float("nan"), clusters

    if n_valid == 1:
        clusters[valid] = 1
        return 1.0, clusters

    # Pairwise RMSD, without structural alignment
    valid_structs = structs[valid]
    coords = valid_structs.reshape(n_valid, -1)

    rmsd = pdist(coords, metric="euclidean") / np.sqrt(L)

    # Catch numerical overflow
    if not np.isfinite(rmsd).all():
        return float("nan"), clusters

    Z = linkage(rmsd, method="single")
    valid_clusters = fcluster(
        Z, t=th, criterion="distance"
    )

    clusters[valid] = valid_clusters

    return len(np.unique(valid_clusters)) / n_valid, clusters


def diversity(seqs: List[str], structs: np.ndarray):
    seq_div, seq_clu = seq_diversity(seqs)

    if structs is None:
        return seq_div, None, seq_div, None

    if len(seqs) != len(structs):
        raise ValueError(
            "seqs and structs must have the same number of candidates"
        )

    struct_div, struct_clu = struct_diversity(structs)

    # Only use valid structures for consistency
    valid = struct_clu > 0

    if not valid.any():
        return seq_div, struct_div, float("nan"), float("nan")

    co_div = np.sqrt(seq_div * struct_div)

    seq_labels = seq_clu[valid]
    struct_labels = struct_clu[valid]

    # Remap cluster labels to contiguous indices
    _, seq_idx = np.unique(seq_labels, return_inverse=True)
    _, struct_idx = np.unique(struct_labels, return_inverse=True)

    n_seq_clu = len(np.unique(seq_idx))
    n_struct_clu = len(np.unique(struct_idx))

    if n_seq_clu == 1 or n_struct_clu == 1:
        consistency = (
            1.0 if n_seq_clu == n_struct_clu else 0.0
        )
    else:
        table = np.zeros(
            (n_seq_clu, n_struct_clu), dtype=int
        )

        np.add.at(table, (seq_idx, struct_idx), 1)

        consistency = association(
            table, method="cramer"
        )

    return seq_div, struct_div, co_div, consistency
