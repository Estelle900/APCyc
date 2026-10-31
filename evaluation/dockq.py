#!/usr/bin/env python3

import os
import sys
import warnings

from globals import DOCKQ_DIR

_DOCKQ_DIR = os.path.abspath(os.fspath(DOCKQ_DIR))
if _DOCKQ_DIR not in sys.path:
    sys.path.insert(0, _DOCKQ_DIR)

from DockQ import calc_DockQ, get_pdb_chains, make_two_chain_pdb_perm  # noqa: E402


class DockQScorer:
    """Reuse an input native PDB across all candidates for one binding pocket.

    A prepared/renumbered native is only needed if a model or native has more
    than two chains. It is created lazily, once per scorer, then deleted at exit.
    """

    def __init__(self, native_pdb: str, pep_chain: str):
        self.native_pdb = os.fspath(native_pdb)
        self.pep_chain = pep_chain
        self.native_chains = get_pdb_chains(self.native_pdb)
        self._prepared_native = None
        self._closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        if not self._closed:
            self._closed = True
            if self._prepared_native is not None:
                try:
                    os.remove(self._prepared_native)
                except FileNotFoundError:
                    pass
                self._prepared_native = None

    def _native_for_multichain(self):
        if self._prepared_native is None:
            other = [chain for chain in self.native_chains if chain != self.pep_chain]
            self._prepared_native = make_two_chain_pdb_perm(
                self.native_pdb, [self.pep_chain], other
            )
        return self._prepared_native

    def score(self, mod_pdb: str) -> float:
        """Score one candidate; return the same 3-decimal output as old CLI."""
        if self._closed:
            raise RuntimeError('DockQScorer is already closed')

        model_chains = get_pdb_chains(mod_pdb)
        if len(model_chains) < 2 or len(self.native_chains) < 2:
            return float('nan')

        if len(model_chains) == 2 and len(self.native_chains) == 2:
            value = calc_DockQ(mod_pdb, self.native_pdb)['DockQ']
        else:
            if (self.pep_chain not in model_chains or
                    self.pep_chain not in self.native_chains):
                return float('nan')

            native = self._native_for_multichain()
            other = [chain for chain in model_chains if chain != self.pep_chain]
            model = make_two_chain_pdb_perm(mod_pdb, [self.pep_chain], other)
            try:
                # -no_needle: renumber the chains but do not run sequence alignment.
                value = calc_DockQ(model, native)['DockQ']
            finally:
                os.remove(model)

        return value


def dockq(mod_pdb: str, native_pdb: str, pep_chain: str) -> float:
    """Compatibility wrapper; prefer one DockQScorer per cal_metrics group."""
    with DockQScorer(native_pdb, pep_chain) as scorer:
        return scorer.score(mod_pdb)
