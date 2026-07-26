"""
study1_bvalue_direction_crossing.py
Study 1 (report section 5): crossing b-values with diffusion-direction count.

Every b-value scheme (31: baseline 45 s/mm2 + 1..5 higher shells) is crossed with
five direction counts (6, 10, 12, 24, all) = 155 protocols. Directions are taken
in a fixed reordered sequence (custom_order) so each subset samples the sphere as
uniformly as possible. For each protocol it records mean FA/MD, RMSE vs the full
acquisition, the CRLB on FA/MD, and Monte-Carlo detectability.

Output: bvalue_directions_monte_carlo.xlsx.
"""

import os
import subprocess
from itertools import combinations

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
from scipy.linalg import inv

# =========================
# INPUT FILES
# =========================
DWI  = "dwi.nii"
BVAL = "dwi.bval"
BVEC = "dwi.bvec"
MASK = "mask.nii.gz"

WORKDIR = "fsl_tmp_joint"
os.makedirs(WORKDIR, exist_ok=True)

BASELINE_B = 45

# =========================
# LOAD DATA
# =========================
dwi_img = nib.load(DWI)
data = dwi_img.get_fdata()
affine = dwi_img.affine
mask = nib.load(MASK).get_fdata() > 0
bvals = np.loadtxt(BVAL)
bvecs = np.loadtxt(BVEC)
if bvecs.shape[0] != 3:
    bvecs = bvecs.T


# =========================
# GLOBAL b0 SNR (report section 5.1: mean / SD of the baseline signal in the ROI)
# =========================
def estimate_global_b0_snr(data, bvals, mask):
    b0_idx = np.where(bvals < 50)[0]            # baseline (b ~ 45 s/mm2) volumes
    if len(b0_idx) == 0:
        raise RuntimeError("No baseline (b ~ 45) volumes found.")
    roi_vals = data[..., b0_idx][mask]          # baseline signal within the ROI
    snr = roi_vals.mean() / roi_vals.std()
    print(f"Global baseline SNR (ROI mean/SD): {snr:.2f}")
    return snr


GLOBAL_SNR = estimate_global_b0_snr(data, bvals, mask)


# =========================
# CRLB FOR THE TENSOR (and FA, MD by the delta method)
# =========================
def dti_crlb(gtab_bvals, gtab_bvecs, snr, D_known):
    """Return (crlb_D[6], crlb_FA, crlb_MD). FA uses an analytic Jacobian of
    FA = sqrt(1.5 * Q / P) with P, Q the tensor invariants."""
    bvals = np.asarray(gtab_bvals)
    bvecs = np.asarray(gtab_bvecs)
    sigma2 = 1.0 / (snr**2) if snr > 0 else 1e-6

    # Fisher information for p = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz]
    I = np.zeros((6, 6))
    for i in range(len(bvals)):
        if bvals[i] < 50:
            continue
        g = bvecs[:, i].reshape(3, 1)
        S = np.exp(-bvals[i] * (g.T @ D_known @ g).item())
        gg = g @ g.T
        dS = bvals[i] * S * np.array([gg[0, 0], gg[1, 1], gg[2, 2],
                                      2 * gg[0, 1], 2 * gg[0, 2], 2 * gg[1, 2]])
        I += np.outer(dS, dS) / sigma2

    if not np.all(np.isfinite(I)):
        return np.zeros(6), np.nan, np.nan
    try:
        C = inv(I)
    except (np.linalg.LinAlgError, ValueError):
        C = np.linalg.pinv(I)
    crlb_D = np.sqrt(np.maximum(np.diag(C), 0))

    Dxx, Dyy, Dzz = D_known[0, 0], D_known[1, 1], D_known[2, 2]
    Dxy, Dxz, Dyz = D_known[0, 1], D_known[0, 2], D_known[1, 2]

    # MD = trace/3 (exact gradient, keeps covariance cross-terms)
    J_md = np.array([1/3, 1/3, 1/3, 0.0, 0.0, 0.0])
    crlb_md = np.sqrt(np.maximum(J_md @ C @ J_md, 0))

    # FA from invariants P, Q, T
    P = Dxx**2 + Dyy**2 + Dzz**2 + 2 * (Dxy**2 + Dxz**2 + Dyz**2)
    T = Dxx + Dyy + Dzz
    Q = P - T**2 / 3.0
    if P <= 0 or Q <= 0:                    # isotropic/degenerate -> FA gradient undefined
        return crlb_D, np.nan, crlb_md
    dP = np.array([2*Dxx, 2*Dyy, 2*Dzz, 4*Dxy, 4*Dxz, 4*Dyz])
    dT = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    dQ = dP - (2.0 * T / 3.0) * dT
    J_fa = np.sqrt(1.5) * (dQ * P - Q * dP) / (2.0 * P**1.5 * np.sqrt(Q))
    crlb_fa = np.sqrt(np.maximum(J_fa @ C @ J_fa, 0))
    return crlb_D, crlb_fa, crlb_md


# =========================
# MONTE-CARLO POWER (per-subject SD = CRLB alone; report section 5.1)
# =========================
def monte_carlo_power_dti(n_subjects=10, delta_fa=0.06, delta_md=0.00012,
                          crlb_fa=None, crlb_md=None, n_sim=1000, alpha=0.05):
    """Power to detect the group difference for one protocol. The per-subject SD
    is that protocol's CRLB alone, so between-subject variability is excluded and
    these are best-case powers. Groups are centred on baseline means offset by
    +/- delta/2; FA is clipped to [0, 1] and MD to positive; each of 1000
    iterations uses a Welch t-test at alpha."""
    if not np.isfinite(crlb_fa) or not np.isfinite(crlb_md):
        return np.nan, np.nan

    fa_baseline, md_baseline = 0.53, 0.00088
    fa_detected = md_detected = 0
    for _ in range(n_sim):
        fa_h = np.clip(np.random.normal(fa_baseline + delta_fa/2, crlb_fa, n_subjects), 0, 1)
        fa_d = np.clip(np.random.normal(fa_baseline - delta_fa/2, crlb_fa, n_subjects), 0, 1)
        md_h = np.clip(np.random.normal(md_baseline - delta_md/2, crlb_md, n_subjects), 0, None)
        md_d = np.clip(np.random.normal(md_baseline + delta_md/2, crlb_md, n_subjects), 0, None)
        _, p_fa = stats.ttest_ind(fa_h, fa_d, equal_var=False)
        _, p_md = stats.ttest_ind(md_h, md_d, equal_var=False)
        fa_detected += p_fa < alpha
        md_detected += p_md < alpha
    return fa_detected / n_sim, md_detected / n_sim


# =========================
# FULL-PROTOCOL DTI FIT (reference)
# =========================
full_prefix = os.path.join(WORKDIR, "full")
subprocess.run(["dtifit", "-k", DWI, "-o", full_prefix, "-m", MASK,
                "-r", BVEC, "-b", BVAL, "--wls"], check=True)

fa_full = nib.load(full_prefix + "_FA.nii.gz").get_fdata()
md_full = nib.load(full_prefix + "_MD.nii.gz").get_fdata()
l1 = nib.load(full_prefix + "_L1.nii.gz").get_fdata()
l2 = nib.load(full_prefix + "_L2.nii.gz").get_fdata()
l3 = nib.load(full_prefix + "_L3.nii.gz").get_fdata()
v1 = nib.load(full_prefix + "_V1.nii.gz").get_fdata()
v2 = nib.load(full_prefix + "_V2.nii.gz").get_fdata()
v3 = nib.load(full_prefix + "_V3.nii.gz").get_fdata()

# reference tensor = median of up to 500 sampled ROI voxels
voxel_indices = np.argwhere(mask)
sample = voxel_indices[np.random.choice(len(voxel_indices), min(500, len(voxel_indices)), replace=False)]
tensor_list = []
for idx in sample:
    idx = tuple(idx)
    D = (l1[idx] * np.outer(v1[idx], v1[idx]) +
         l2[idx] * np.outer(v2[idx], v2[idx]) +
         l3[idx] * np.outer(v3[idx], v3[idx]))
    if np.all(np.isfinite(D)):
        tensor_list.append(D)
if not tensor_list:
    raise RuntimeError("No valid tensors found.")
D_ref = np.median(np.stack(tensor_list), axis=0)
print(f"Reference tensor from {len(tensor_list)} voxels.")

# =========================
# B-VALUE SCHEMES (baseline + 1..k higher shells) and DIRECTION COUNTS
# =========================
other_bvals = [b for b in np.unique(bvals) if b != BASELINE_B]
bvalue_combos = sorted({tuple(sorted([BASELINE_B] + list(combo)))
                        for k in range(1, len(other_bvals) + 1)
                        for combo in combinations(other_bvals, k)})
print("b-value schemes:", len(bvalue_combos))

dir_settings = [6, 10, 12, 24, "all"]
# fixed direction order so each subset samples the sphere uniformly
custom_order = [32, 58, 29, 25, 10, 26, 30, 52, 45, 18, 17, 1,
                33, 19, 3, 48, 11, 2, 16, 4, 6, 43, 7, 23]

# =========================
# MAIN SWEEP: each b-value scheme x each direction count
# =========================
rows = []
for combo in bvalue_combos:
    combo_name = "-".join(str(int(b)) for b in combo)
    combo_idx = np.where(np.isin(bvals, combo))[0]
    combo_b0_idx = combo_idx[bvals[combo_idx] <= 50]
    combo_dwi_idx = combo_idx[bvals[combo_idx] > 50]

    for d in dir_settings:
        if d == "all":
            used_dwi = combo_dwi_idx
            dir_label = "all"
        else:
            if len(combo_dwi_idx) < d:
                continue
            positions = [p for p in custom_order if 1 <= p <= len(combo_dwi_idx)][:d]
            used_dwi = np.array([combo_dwi_idx[p - 1] for p in positions])
            dir_label = f"{d}dirs"

        # build and save this protocol's subset
        config_idx = np.sort(np.concatenate([combo_b0_idx, used_dwi]))
        subset_bvals = bvals[config_idx]
        subset_bvecs = bvecs[:, config_idx]
        tmp_dir = os.path.join(WORKDIR, f"{combo_name}_{dir_label}")
        os.makedirs(tmp_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(data[..., config_idx], affine), os.path.join(tmp_dir, "dwi.nii.gz"))
        np.savetxt(os.path.join(tmp_dir, "bval"), subset_bvals.reshape(1, -1), fmt="%.0f")
        np.savetxt(os.path.join(tmp_dir, "bvec"), subset_bvecs, fmt="%.8f")

        subprocess.run(["dtifit", "-k", os.path.join(tmp_dir, "dwi.nii.gz"),
                        "-o", os.path.join(tmp_dir, "dtifit"), "-m", MASK,
                        "-r", os.path.join(tmp_dir, "bvec"),
                        "-b", os.path.join(tmp_dir, "bval"), "--wls"], check=True)

        fa = nib.load(os.path.join(tmp_dir, "dtifit_FA.nii.gz")).get_fdata()
        md = nib.load(os.path.join(tmp_dir, "dtifit_MD.nii.gz")).get_fdata()
        fa_mean, md_mean = float(np.mean(fa[mask])), float(np.mean(md[mask]))
        fa_rmse = np.sqrt(np.mean((fa[mask] - fa_full[mask]) ** 2))
        md_rmse = np.sqrt(np.mean((md[mask] - md_full[mask]) ** 2))

        # SNR scales as sqrt(directions kept / directions available)
        n_directions = len(used_dwi)
        if len(combo_dwi_idx) > 0 and np.isfinite(GLOBAL_SNR):
            subset_snr = GLOBAL_SNR * np.sqrt(n_directions / len(combo_dwi_idx))
        else:
            subset_snr = 0.0
        if subset_snr < 5:
            print(f"Warning: low SNR ({subset_snr:.2f}) for {combo_name} {dir_label}")

        _, crlb_fa, crlb_md = dti_crlb(subset_bvals, subset_bvecs, subset_snr, D_ref)
        fa_power, md_power = monte_carlo_power_dti(
            n_subjects=10, delta_fa=0.06, delta_md=0.00012, crlb_fa=crlb_fa, crlb_md=crlb_md)
        min_power = np.nanmin([fa_power, md_power])

        rows.append([combo_name, dir_label, subset_snr, fa_mean, md_mean,
                     fa_rmse, md_rmse, crlb_fa, crlb_md, fa_power, md_power, min_power])
        print(f"{combo_name:20s}{dir_label:8s}SNR={subset_snr:6.2f}"
              f"  FA power={fa_power:.3f}  MD power={md_power:.3f}")

# =========================
# EXCEL OUTPUT
# =========================
df = pd.DataFrame(rows, columns=["Combo", "Directions", "SNR", "FA_Mean", "MD_Mean",
                                 "RMSE_FA", "RMSE_MD", "CRLB_FA", "CRLB_MD",
                                 "FA_Power", "MD_Power", "Min_Power"])
df.to_excel("bvalue_directions_monte_carlo.xlsx", index=False)
print("Saved bvalue_directions_monte_carlo.xlsx")
