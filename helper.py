"""Read the acquisition table and convert b-values to q for the CTRW model."""

import math

# Diffusion times (s) for the five blocks; gradient pulse duration delta = 10 ms.
DELTA_BIG = [0.070, 0.130, 0.190, 0.250, 0.330]
DELTA_SMALL = 0.010
B0_MARKERS = [15, 30, 45, 65, 85]        # nominal reference b-value per block


def bval_to_q(b, Delta, delta):
    """b-value (s/mm^2) -> q (mm^-1):  q = sqrt(b / (Delta - delta/3)) / (2*pi)."""
    if b == 0:
        return 0.0
    return math.sqrt(b / (Delta - delta / 3.0)) / (2.0 * math.pi)


def read_acqs(acqfilename):
    """Return [q, Delta] for each volume in the acquisition file. A new diffusion-
    time block starts at each low-b (< 100 s/mm^2) reference volume."""
    with open(acqfilename) as f:
        lines = f.readlines()

    qDels = []
    block_index = -1
    Delta = None
    for line in lines:
        bval = float(line.split()[3])
        if int(round(bval)) < 100:
            block_index += 1
            Delta = DELTA_BIG[block_index]
            q = 0.0
        else:
            q = bval_to_q(bval, Delta, DELTA_SMALL)
        qDels.append([q, Delta])

    print(f"Read {len(qDels)} acquisitions across {block_index + 1} diffusion times")
    return qDels
