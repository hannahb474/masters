"""
dti_scheme.py
Parse a multi-diffusion-time DWI acquisition (shared bval/bvec across subjects)
and return the exact volume indices for any subset of diffusion times and
b-values, i.e. build a shortened scan from the full acquisition.

Acquisition this targets: 5 diffusion times (70, 130, 190, 250, 330 ms),
b-values 0/200/400/600/800 s/mm^2 with 1/4/5/6/6 averages and 6 directions per
shell. The b=0 of each block is a small diffusion-time-dependent reference
(~15/30/45/65/85). Blocks are detected from the low-b reference volumes and the
shells are read from the bval file, so nothing about the geometry is hard-coded.
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Block:
    """One diffusion-time acquisition (one Delta)."""
    delta: float
    b0_index: int
    b0_value: float
    start: int
    stop: int
    shells: dict = field(default_factory=dict)   # {nominal_b: [volume indices]}


class AcquisitionScheme:
    def __init__(self, bvals, bvecs, diffusion_times,
                 b0_threshold=150.0, shell_tol=25.0):
        self.bvals = np.asarray(bvals, dtype=float).ravel()
        bvecs = np.asarray(bvecs, dtype=float)
        if bvecs.shape[0] != 3:
            bvecs = bvecs.T
        self.bvecs = bvecs
        self.N = self.bvals.size
        if self.bvecs.shape[1] != self.N:
            raise ValueError(f"bvec/bval length mismatch: {self.bvecs.shape} vs {self.N}")

        self.b0_threshold = b0_threshold
        self.shell_tol = shell_tol
        self.diffusion_times = list(diffusion_times)
        self.blocks = self._detect_blocks()
        self.nominal_bvalues = self._detect_nominal_shells()

    def _detect_blocks(self):
        # each diffusion-time block begins at a low-b reference volume
        b0_idx = np.where(self.bvals < self.b0_threshold)[0]
        if len(b0_idx) != len(self.diffusion_times):
            raise ValueError(
                f"Found {len(b0_idx)} reference (b<{self.b0_threshold}) volumes "
                f"but {len(self.diffusion_times)} diffusion times were given.")
        starts = list(b0_idx)
        stops = list(b0_idx[1:]) + [self.N]
        blocks = []
        for delta, s, stop in zip(self.diffusion_times, starts, stops):
            blk = Block(delta=delta, b0_index=s, b0_value=float(self.bvals[s]),
                        start=s, stop=stop)
            for i in range(s, stop):
                b = self.bvals[i]
                if b < self.b0_threshold:
                    continue
                nominal = int(round(b / 100.0) * 100)     # snap to nearest 100
                blk.shells.setdefault(nominal, []).append(i)
            blocks.append(blk)
        return blocks

    def _detect_nominal_shells(self):
        shells = set()
        for blk in self.blocks:
            shells.update(blk.shells.keys())
        return sorted(shells)

    def select_indices(self, delta_subset, bvalue_subset, include_b0=True):
        """Volume indices for a shortened scan. Returns a dict keyed by delta:
        {delta: {'indices': [...], 'b0_index': int, 'shells': {b: [...]}}}."""
        delta_subset = set(delta_subset)
        bvalue_subset = set(int(b) for b in bvalue_subset)
        out = {}
        for blk in self.blocks:
            if blk.delta not in delta_subset:
                continue
            idx = [blk.b0_index] if include_b0 else []
            kept_shells = {}
            for b in sorted(blk.shells):
                if b in bvalue_subset:
                    kept_shells[b] = list(blk.shells[b])
                    idx.extend(blk.shells[b])
            out[blk.delta] = {"indices": sorted(idx), "b0_index": blk.b0_index,
                              "shells": kept_shells}
        return out

    def subset_gtab_arrays(self, indices):
        """(bvals, bvecs) sliced to `indices`."""
        indices = list(indices)
        return self.bvals[indices].copy(), self.bvecs[:, indices].copy()

    def volume_count(self, selection):
        """Total acquired volumes in a selection (proxy for scan time)."""
        return sum(len(v["indices"]) for v in selection.values())

    def full_volume_count(self):
        return self.N

    def enumerate_protocols(self, min_deltas=1, min_bvalues=1, require_shells_for_fit=1):
        """Yield every (delta_subset, bvalue_subset) combination. At least
        `require_shells_for_fit` non-zero shells are kept so a tensor can be fit."""
        d_subsets = []
        for r in range(min_deltas, len(self.diffusion_times) + 1):
            d_subsets.extend(itertools.combinations(self.diffusion_times, r))
        b_subsets = []
        for r in range(max(min_bvalues, require_shells_for_fit), len(self.nominal_bvalues) + 1):
            b_subsets.extend(itertools.combinations(self.nominal_bvalues, r))
        for d in d_subsets:
            for b in b_subsets:
                yield d, b


def load_scheme(bval_path, bvec_path, diffusion_times=(70, 130, 190, 250, 330), **kw):
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    return AcquisitionScheme(bvals, bvecs, diffusion_times, **kw)
