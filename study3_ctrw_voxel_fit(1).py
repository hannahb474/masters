"""
study3_ctrw_voxel_fit.py
Study 3 (report section 7.1): fit the CTRW (space-time fractional / Mittag-Leffler)
model voxelwise for one subject.

The model is isotropic, so the per-direction signal is first averaged to one value
per (q, Delta) shell, then fit for theta, alpha and beta (D is derived). Only
voxels inside a muscle mask are fit; at-bound fits (alpha=1 / beta=2, i.e. Gaussian
diffusion) are kept and flagged in ok_map rather than discarded.

Run:  python study3_ctrw_voxel_fit.py <subject>/dwi.nii
Saves theta / al / bt / S0 / ok / D maps as <prefix>_map_<name>.nii.gz.

Requires local modules helper.py, average_shells.py and ml3.py (the fractional
model; the original CTRW/Mittag-Leffler code, adapted here).
"""

import os
import sys
import time as _time

import numpy as np
import nibabel as nib
from joblib import Parallel, delayed

import helper
from average_shells import average_by_shell
import ml3                                    # 3-parameter Mittag-Leffler model

base_path = os.environ.get("DATA_ROOT", "data")   # folder containing the per-subject directories

# --- masking: threshold a b0 volume at corner_noise_mean + NOISE_SIGMA * SD ---
NOISE_SIGMA = 5.0
CORNER = 8            # size of the corner squares used to estimate air noise
MASK_FILE = None      # set to a segmentation (in the subject folder) to use it instead

# =========================
# RESOLVE PATHS FROM THE INPUT IMAGE
# =========================
if len(sys.argv) >= 2:
    imagefilename = sys.argv[1]
else:
    imagefilename = 'subject_01/dwi.nii'
    print(f"WARNING: no argument supplied, defaulting to {imagefilename}")

dogname = imagefilename[:imagefilename.index('.')]
subject_folder = imagefilename.split('/')[0]
data_path = os.path.join(base_path, subject_folder)
acqfilename = os.path.join(base_path, subject_folder, 'rawdata_dirs.txt')
imagepath = os.path.join(base_path, imagefilename)

# =========================
# LOAD IMAGE + ACQUISITION TABLE
# =========================
print(f"Loading image '{imagefilename}'")
img = nib.load(imagepath)
data = img.get_fdata()
dims3 = data.shape[:3]
print("dims:", data.shape)

qDels_raw = helper.read_acqs(acqfilename)          # (q, Delta) per volume
_qd = np.asarray(qDels_raw, dtype=float)
if _qd.shape[0] != data.shape[3]:
    raise SystemExit(f"Acquisition count mismatch: {_qd.shape[0]} entries in "
                     f"{acqfilename} vs {data.shape[3]} volumes in the image.")

# output maps (theta is the comparable decay parameter; S0!=0 marks a produced fit)
theta_map = np.zeros(dims3, float)
al_map    = np.zeros(dims3, float)
bt_map    = np.zeros(dims3, float)
S0_map    = np.zeros(dims3, float)
ok_map    = np.zeros(dims3, float)   # 1 = converged and NOT on a parameter bound
D_map     = np.zeros(dims3, float)   # derived; units vary per voxel (reference only)

# reference scales for theta = D * q0^beta * Del0^alpha (must match across subjects)
_q_nz = _qd[:, 0][_qd[:, 0] > 0]
SCALES = ml3.ML3Scales(q0=float(np.median(_q_nz)), Del0=float(np.median(_qd[:, 1])))
print(f"ML3Scales: q0={SCALES.q0:.6g}, Del0={SCALES.Del0:.6g}  (must match across subjects)")


# =========================
# VOXEL MASK
# =========================
def build_mask():
    """Muscle mask: a segmentation if MASK_FILE is set, else a b0 volume
    thresholded against noise measured in the image corners (air)."""
    if MASK_FILE:
        m = nib.load(os.path.join(data_path, MASK_FILE)).get_fdata() > 0
        if m.shape != dims3:
            raise SystemExit(f"Mask shape {m.shape} != image {dims3}")
        return m

    qs = _qd[:, 0]
    b0_idx = np.where(qs <= 0)[0]
    if len(b0_idx) == 0:
        b0_idx = np.array([int(np.argmin(qs))])
        print(f"  WARNING: no q==0 volume; using lowest-q volume {b0_idx[0]} as b0")
    b0 = data[..., b0_idx].mean(axis=-1)

    c = max(1, min(CORNER, dims3[0] // 2, dims3[1] // 2))
    corner = np.concatenate([b0[:c, :c].ravel(), b0[-c:, -c:].ravel(),
                             b0[:c, -c:].ravel(), b0[-c:, :c].ravel()])
    mu, sd = float(corner.mean()), float(corner.std())
    thresh = mu + NOISE_SIGMA * sd
    if not np.isfinite(thresh) or thresh <= 0:
        thresh = 0.05 * float(np.nanmax(b0))
    print(f"  corner noise mean={mu:.2f} sd={sd:.2f} -> threshold={thresh:.2f}")
    return b0 > thresh


fit_mask = build_mask()
voxels_to_fit = [tuple(i) for i in np.argwhere(fit_mask)]
n_total = int(np.prod(dims3))
n_to_fit = len(voxels_to_fit)
print(f"  {n_to_fit}/{n_total} voxels in mask ({n_to_fit / n_total:.1%})")
if n_to_fit == 0:
    raise SystemExit("Mask is empty -- check the threshold or supply MASK_FILE.")


# =========================
# PER-VOXEL FIT
# =========================
def _shell_weights(n_avg):
    """sqrt(n_avg) shell weights, or None (unweighted) if counts are unusable."""
    try:
        n = np.asarray(n_avg, dtype=float)
        if n.ndim == 1 and np.all(np.isfinite(n)) and np.all(n > 0):
            return ml3.averages_to_weights(n)
    except (TypeError, ValueError):
        pass
    return None


def fit_voxel(signal_, qDels_raw_, strict=False):
    """Fit one voxel -> (theta, alpha, beta, S0, ok). Directions are averaged to
    (q, Delta) shells first. At-bound fits keep their parameters (ok=0); only a
    numerical failure returns zeros. strict=True re-raises (used on a probe voxel)."""
    try:
        shell_qDels, shell_signal, n_avg = average_by_shell(qDels_raw_, signal_)
        theta, al, bt, S0, ok = ml3.fit_voxel(
            shell_qDels, shell_signal, SCALES, weights=_shell_weights(n_avg))
        if not all(np.isfinite([theta, al, bt, S0])):
            return (0.0, 0.0, 0.0, 0.0, 0)
        return (float(theta), float(al), float(bt), float(S0), int(bool(ok)))
    except Exception:
        if strict:
            raise
        return (0.0, 0.0, 0.0, 0.0, 0)


# fail fast: probe one voxel with a real traceback before launching the workers
_probe_idx = voxels_to_fit[len(voxels_to_fit) // 2]
_p = fit_voxel(data[_probe_idx], qDels_raw, strict=True)
print(f"Probe voxel {_probe_idx}: theta={_p[0]:.4g} alpha={_p[1]:.4g} "
      f"beta={_p[2]:.4g} S0={_p[3]:.4g} ok={_p[4]}")

# parallel fit, reporting progress between chunks (the driver parses PROGRESS lines)
print(f"Fitting {n_to_fit} voxels...")
fit_start = _time.time()
CHUNK = max(1, n_to_fit // 50)
results = []
for start in range(0, n_to_fit, CHUNK):
    chunk = voxels_to_fit[start:start + CHUNK]
    results.extend(Parallel(n_jobs=-1)(delayed(fit_voxel)(data[i], qDels_raw) for i in chunk))
    print(f"PROGRESS: {len(results)}/{n_to_fit} elapsed={_time.time() - fit_start:.1f}", flush=True)

for idx, (theta, al, bt, S0, ok) in zip(voxels_to_fit, results):
    theta_map[idx], al_map[idx], bt_map[idx], S0_map[idx], ok_map[idx] = theta, al, bt, S0, ok


# =========================
# VOXEL FUNNEL (coverage diagnostics)
# =========================
fitted = S0_map != 0.0
n_fit = int(fitted.sum())
n_ok = int((ok_map == 1).sum())
print("\n---- voxel funnel ----")
print(f"  in image                : {n_total}")
print(f"  in mask                 : {n_to_fit}")
print(f"  fit produced            : {n_fit}")
print(f"  converged, NOT at bound : {n_ok}")
print(f"  converged, AT a bound   : {n_fit - n_ok}  (saved with ok=0; alpha=1/beta=2 is Gaussian)")

if n_fit == 0:
    raise SystemExit("Zero voxels produced a fit -- refusing to write all-zero maps.")
if n_fit / max(n_to_fit, 1) < 0.5:
    print("  WARNING: >half of masked voxels failed -- check shell averaging and the b0 threshold.")

# derived D (units vary per voxel -> reference only; compare theta between groups)
D_map[fitted] = SCALES.theta_to_D(theta_map[fitted], al_map[fitted], bt_map[fitted])


# =========================
# SAVE MAPS
# =========================
base_name = os.path.basename(dogname)

def _save(arr, prefix):
    hdr = img.header.copy()
    hdr.set_data_dtype(np.float32)
    hdr['scl_slope'], hdr['scl_inter'] = 1.0, 0.0
    im = nib.Nifti1Image(arr.astype(np.float32), img.affine, header=hdr)
    im.set_data_dtype(np.float32)
    path = os.path.join(data_path, f"{prefix}_map_{base_name}.nii.gz")
    nib.save(im, path)
    print("  saved", path)

for arr, prefix in [(theta_map, "theta"), (al_map, "al"), (bt_map, "bt"),
                    (S0_map, "S0"), (ok_map, "ok"), (D_map, "D")]:
    _save(arr, prefix)
print("Done.")
