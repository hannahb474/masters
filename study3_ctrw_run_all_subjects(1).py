"""
study3_ctrw_run_all_subjects.py
Study 3 (report section 7.2): driver for the CTRW fit.

Runs study3_ctrw_voxel_fit.py on every subject, then pools each subject's
theta/alpha/beta/D maps within the muscle mask into per-subject summary
statistics and writes them to Excel (all_subjects / healthy / DMD / summary
sheets). Feeds study3_group_comparison.py.

Notes:
  * theta (dimensionless) is the comparable decay parameter; raw D has
    voxel-dependent units and is kept for reference only.
  * The fitted-voxel mask is S0 != 0 (a fit was produced), NOT alpha != 0, so
    at-bound (Gaussian) voxels are retained. Every parameter is summarised over
    all fitted voxels AND over the ok-only subset (ok_* columns); a gap between
    them means the at-bound filter is materially changing the result.
"""

import os
import sys
import re
import subprocess
import time

import numpy as np
import nibabel as nib
import pandas as pd

# =========================
# SETTINGS
# =========================
BASE = os.environ.get("DATA_ROOT", "data")   # folder containing the per-subject directories
# Subject folder names under DATA_ROOT (edit to match your dataset).
HEALTHY  = ['control_01', 'control_02', 'control_03', 'control_04', 'control_05']
PATIENTS = ['patient_01', 'patient_02', 'patient_03', 'patient_04']
ALL_SUBJECTS = HEALTHY + PATIENTS

FRAC_FITTER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'study3_ctrw_voxel_fit.py')
RESULTS_XLSX = os.path.join(BASE, 'ctrw_fitting_results.xlsx')

PARAMS = ['theta', 'alpha', 'beta']          # D is summarised too, for reference only
PREFIX = {'theta': 'theta', 'alpha': 'al', 'beta': 'bt', 'S0': 'S0', 'ok': 'ok', 'D': 'D'}


def format_time(s):
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s//60)}m {int(s%60)}s"
    return f"{int(s//3600)}h {int((s%3600)//60)}m"


PROGRESS_RE = re.compile(r"PROGRESS:\s*(\d+)/(\d+)\s+elapsed=([\d.]+)")


def run_fitter(subject):
    """Run the voxel fitter for one subject, echoing its live PROGRESS bar."""
    proc = subprocess.Popen([sys.executable, FRAC_FITTER, f"{subject}/dwi.nii"],
                            cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        lines.append(line)
        m = PROGRESS_RE.search(line)
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            print(f"\r    {subject}: {done}/{total} voxels "
                  f"({done/total:5.1%})   ", end="", flush=True)
    proc.wait()
    print()
    return proc.returncode, "".join(lines)


# =========================
# STEP 1: fit every subject
# =========================
print("STEP 1: fitting each subject")
fit_ok, fit_failed, fit_skipped = [], [], []
for subject in ALL_SUBJECTS:
    if not os.path.exists(os.path.join(BASE, subject, 'dwi.nii')):
        print(f"  {subject}: dwi.nii not found -- skipping")
        fit_skipped.append(subject)
        continue
    print(f"\n--- {subject} ---")
    rc, out = run_fitter(subject)
    if rc != 0:
        print(f"  ERROR fitting {subject} (exit {rc}):")
        print("  " + "\n  ".join(out.strip().splitlines()[-25:]))
        fit_failed.append(subject)
    else:
        fit_ok.append(subject)

print(f"\nfitted OK: {len(fit_ok)}  failed: {len(fit_failed)}  skipped: {len(fit_skipped)}")
if fit_failed:
    print(f"  FAILED: {', '.join(fit_failed)}")
if not fit_ok:
    sys.exit("No subject fitted successfully -- read the fitter traceback(s) above.")


# =========================
# STEP 2: masked per-subject statistics
# =========================
print("\nSTEP 2: pooling maps per subject")


def load_maps(subject):
    """Load theta/alpha/beta/S0 (required) and ok/D (optional). None if a required map is missing."""
    maps = {}
    for name in PARAMS + ['S0']:
        path = os.path.join(BASE, subject, f"{PREFIX[name]}_map_dwi.nii.gz")
        if not os.path.exists(path):
            print(f"  {subject}: missing {os.path.basename(path)}")
            return None
        maps[name] = nib.load(path).get_fdata()
    for name in ('ok', 'D'):
        path = os.path.join(BASE, subject, f"{PREFIX[name]}_map_dwi.nii.gz")
        maps[name] = nib.load(path).get_fdata() if os.path.exists(path) else None
    return maps


def masked_stats(arr, mask, prefix, out):
    """mean / std / median of finite arr[mask] into out[prefix_*]."""
    if arr is None or mask is None or mask.sum() == 0:
        out[f'{prefix}_mean'] = out[f'{prefix}_std'] = out[f'{prefix}_median'] = np.nan
        return
    v = arr[mask]
    v = v[np.isfinite(v)]
    if v.size == 0:
        out[f'{prefix}_mean'] = out[f'{prefix}_std'] = out[f'{prefix}_median'] = np.nan
    else:
        out[f'{prefix}_mean'] = v.mean()
        out[f'{prefix}_std'] = v.std()
        out[f'{prefix}_median'] = np.median(v)


rows = []
for subject in ALL_SUBJECTS:
    maps = load_maps(subject)
    if maps is None:
        print(f"  {subject}: maps not loadable -- skipping")
        continue

    S0 = maps['S0']
    mask = S0 != 0                                   # a fit was produced (keeps at-bound voxels)
    if mask.sum() == 0:
        print(f"  {subject}: no fitted voxels -- skipping")
        continue
    ok_mask = (maps['ok'] > 0.5) & mask if maps['ok'] is not None else None

    row = {
        'subject':  subject,
        'group':    'healthy' if subject in HEALTHY else 'DMD',
        'n_voxels': int(mask.sum()),
        'n_ok':     int(ok_mask.sum()) if ok_mask is not None else np.nan,
    }
    row['frac_at_bound'] = (1.0 - row['n_ok'] / row['n_voxels']) if ok_mask is not None else np.nan

    # stats over all fitted voxels, and over the ok-only subset
    for name in PARAMS:
        masked_stats(maps[name], mask, name, row)
    masked_stats(maps['D'], mask, 'D', row)          # reference only -- units vary
    for name in PARAMS:
        masked_stats(maps[name], ok_mask, f'ok_{name}', row)

    # per-slice stats (all fitted voxels)
    for sl in range(S0.shape[2]):
        sl_mask = S0[:, :, sl] != 0
        for name in PARAMS + ['D']:
            sub = {}
            if sl_mask.sum() > 0:
                masked_stats(maps[name][:, :, sl], sl_mask, name, sub)
            else:
                sub[f'{name}_mean'] = sub[f'{name}_std'] = np.nan
            row[f'{name}_mean_slice{sl+1}'] = sub[f'{name}_mean']
            row[f'{name}_std_slice{sl+1}'] = sub[f'{name}_std']

    rows.append(row)

print(f"  processed {len(rows)} subjects")
if not rows:
    sys.exit("No subject produced usable maps -- see the 'missing'/'no fitted voxels' messages above.")


# =========================
# STEP 3: write Excel
# =========================
print("\nSTEP 3: writing Excel")
df = pd.DataFrame(rows)

print("\nVoxel coverage:")
print(df[['subject', 'group', 'n_voxels', 'n_ok', 'frac_at_bound']].to_string(index=False))
if df['frac_at_bound'].notna().any() and df['frac_at_bound'].mean() > 0.2:
    print(f"\n  NOTE: on average {df['frac_at_bound'].mean():.0%} of fitted voxels sit on a "
          "bound; compare the *_ columns against ok_* to see if excluding them changes the result.")

h = df[df['group'] == 'healthy']
d = df[df['group'] == 'DMD']

with pd.ExcelWriter(RESULTS_XLSX) as writer:
    df.to_excel(writer, sheet_name='all_subjects', index=False)
    h.to_excel(writer, sheet_name='healthy', index=False)
    d.to_excel(writer, sheet_name='DMD', index=False)

    metrics = ([f'{p}_{s}' for p in PARAMS for s in ('mean', 'std')]
               + [f'ok_{p}_{s}' for p in PARAMS for s in ('mean', 'std')]
               + ['D_mean', 'D_std', 'n_voxels', 'n_ok', 'frac_at_bound'])

    def _note(m):
        if m.startswith('D_'):
            return 'raw D: units vary per voxel -- not comparable; compare theta'
        if m.startswith('ok_'):
            return 'at-bound voxels excluded'
        return ''

    summary = pd.DataFrame({
        'metric':       metrics,
        'healthy_mean': [h[m].mean() if m in h else np.nan for m in metrics],
        'DMD_mean':     [d[m].mean() if m in d else np.nan for m in metrics],
        'note':         [_note(m) for m in metrics],
    })
    summary.to_excel(writer, sheet_name='summary', index=False)

print(f"\nSaved to {RESULTS_XLSX}")
print("Done.")
