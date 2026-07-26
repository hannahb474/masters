"""Average the signal across gradient directions within each (q, Delta) shell.

The CTRW model is isotropic (depends on q and Delta only), so per-direction
signal is first collapsed to one value per shell; otherwise real anisotropy
enters the fit as noise.
"""

import numpy as np


def average_by_shell(qDels, signal, decimals=6):
    """Group acquisitions by (round(q, decimals), Delta) and average the signal
    in each shell. Returns (shell_qDels, shell_signal, shell_counts) in first-seen
    order. `decimals` rounds q so floating-point noise does not split one shell.
    """
    qDels = np.asarray(qDels, dtype=float)
    signal = np.asarray(signal, dtype=float)

    q_rounded = np.round(qDels[:, 0], decimals)
    Delta = qDels[:, 1]
    keys = list(zip(q_rounded, Delta))

    seen, order = {}, []
    for key, s in zip(keys, signal):
        if key not in seen:
            seen[key] = []
            order.append(key)
        seen[key].append(s)

    shell_qDels = [[k[0], k[1]] for k in order]
    shell_signal = np.array([np.mean(seen[k]) for k in order])
    shell_counts = np.array([len(seen[k]) for k in order])
    return shell_qDels, shell_signal, shell_counts
