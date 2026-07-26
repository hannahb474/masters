"""
study3_group_comparison.py
Study 3 (report section 7.2): compare controls vs DMD on the five metrics.

Loads the pooled CTRW parameters (alpha, beta, D) from the driver's Excel, runs
FSL dtifit on the TA mask to add FA and MD per subject, then tests each metric
between groups with BOTH a Mann-Whitney U test and a Welch t-test (Cohen's d,
95% CI, Shapiro-Wilk noted). Produces statistical_results.xlsx (Table 7.3) and a
five-panel box/scatter plot (Figure 7.2).

Caveat: with 5 controls vs 4 patients the smallest two-sided Mann-Whitney p is
0.016, so p<0.05 is near-binary; read effect sizes alongside it.
"""

import os
import subprocess

import numpy as np
import nibabel as nib
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
BASE = os.environ.get("DATA_ROOT", "data")   # folder containing the per-subject directories
# Subject folder names under DATA_ROOT (edit to match your dataset).
HEALTHY  = ['control_01', 'control_02', 'control_03', 'control_04', 'control_05']
PATIENTS = ['patient_01', 'patient_02', 'patient_03', 'patient_04']
ALL_SUBJECTS = HEALTHY + PATIENTS

DWI_FILE, BVAL_FILE, BVEC_FILE, MASK_FILE = 'dwi.nii', 'dwi.bval', 'dwi.bvec', 'mask.nii'
CTRW_XLSX = os.path.join(BASE, 'ctrw_fitting_results.xlsx')   # written by study3_ctrw_run_all_subjects.py

WORKDIR = os.path.join(BASE, 'ta_dtifit_tmp')
os.makedirs(WORKDIR, exist_ok=True)

# =========================
# STEP 1: load pooled CTRW parameters (alpha, beta, D)
# =========================
print("Loading CTRW parameters...")
df_ab = pd.read_excel(CTRW_XLSX, sheet_name='all_subjects')
if 'D_mean' not in df_ab.columns:
    raise RuntimeError("D_mean not found -- run study3_ctrw_run_all_subjects.py first.")
print(f"  {len(df_ab)} subjects")

# =========================
# STEP 2: dtifit on the TA mask -> FA, MD per subject
# =========================
print("\nSTEP 2: dtifit for FA and MD")
dti_rows = []
for subject in ALL_SUBJECTS:
    group = 'healthy' if subject in HEALTHY else 'DMD'
    sp = os.path.join(BASE, subject)
    paths = {k: os.path.join(sp, f) for k, f in
             [('dwi', DWI_FILE), ('bval', BVAL_FILE), ('bvec', BVEC_FILE), ('mask', MASK_FILE)]}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        print(f"  {subject}: skipping -- missing {missing}")
        continue

    out_prefix = os.path.join(WORKDIR, f"{subject}_TA_dtifit")
    subprocess.run(["dtifit", "-k", paths['dwi'], "-o", out_prefix, "-m", paths['mask'],
                    "-r", paths['bvec'], "-b", paths['bval'], "--wls"], check=True)

    fa_path, md_path = out_prefix + "_FA.nii.gz", out_prefix + "_MD.nii.gz"
    if not (os.path.exists(fa_path) and os.path.exists(md_path)):
        print(f"  {subject}: dtifit output missing -- skipping")
        continue

    mask = nib.load(paths['mask']).get_fdata() > 0
    fa = nib.load(fa_path).get_fdata()[mask]
    md = nib.load(md_path).get_fdata()[mask]
    fa, md = fa[np.isfinite(fa)], md[np.isfinite(md)]
    if len(fa) == 0 or len(md) == 0:
        print(f"  {subject}: no finite FA/MD -- skipping")
        continue

    print(f"  {subject} ({group}): FA={fa.mean():.4f}, MD={md.mean():.6f}")
    dti_rows.append({'subject': subject, 'group': group,
                     'FA_mean': fa.mean(), 'FA_std': fa.std(),
                     'MD_mean': md.mean(), 'MD_std': md.std()})

dti_df = pd.DataFrame(dti_rows)
if dti_df.empty:
    raise RuntimeError("No subjects produced valid dtifit results.")

# =========================
# STEP 3: merge CTRW + DTI metrics
# =========================
df = df_ab.merge(dti_df, on=['subject', 'group'], how='inner')
print(f"\nMerged dataset: {len(df)} subjects")
healthy = df[df['group'] == 'healthy']
dmd     = df[df['group'] == 'DMD']

# =========================
# STEP 4: Mann-Whitney U + Welch t-test on each metric
# =========================
print("\nStatistical tests")
metrics = ['alpha_mean', 'beta_mean', 'D_mean', 'FA_mean', 'MD_mean']
mw_results, tt_results = [], []

for metric in metrics:
    h_vals = healthy[metric].dropna()
    d_vals = dmd[metric].dropna()

    # Mann-Whitney U + rank-biserial effect size
    U, p_mw = stats.mannwhitneyu(h_vals, d_vals, alternative='two-sided')
    effect_size = 1 - (2 * U) / (len(h_vals) * len(d_vals))
    mw_results.append({
        'metric': metric,
        'healthy_mean': h_vals.mean(), 'healthy_std': h_vals.std(),
        'DMD_mean': d_vals.mean(), 'DMD_std': d_vals.std(),
        'U_statistic': U, 'p_value': p_mw, 'effect_size': effect_size,
        'significant': p_mw < 0.05,
    })

    # Welch t-test + Cohen's d + 95% CI (Shapiro-Wilk noted, low power at this n)
    _, sh_h = stats.shapiro(h_vals)
    _, sh_d = stats.shapiro(d_vals)
    t_stat, p_tt = stats.ttest_ind(h_vals, d_vals, equal_var=False)
    pooled_std = np.sqrt((h_vals.std()**2 + d_vals.std()**2) / 2)
    cohens_d = (h_vals.mean() - d_vals.mean()) / pooled_std if pooled_std != 0 else 0
    diff = h_vals.mean() - d_vals.mean()
    se = np.sqrt(h_vals.std()**2 / len(h_vals) + d_vals.std()**2 / len(d_vals))
    tt_results.append({
        'metric': metric,
        'healthy_mean': h_vals.mean(), 'healthy_std': h_vals.std(),
        'DMD_mean': d_vals.mean(), 'DMD_std': d_vals.std(),
        't_statistic': t_stat, 'p_value': p_tt, 'cohens_d': cohens_d,
        'CI_95_low': diff - 1.96 * se, 'CI_95_high': diff + 1.96 * se,
        'shapiro_p_healthy': sh_h, 'shapiro_p_DMD': sh_d,
        'normality_healthy': 'normal' if sh_h > 0.05 else 'non-normal',
        'normality_DMD': 'normal' if sh_d > 0.05 else 'non-normal',
        'significant': p_tt < 0.05,
    })

    print(f"  {metric}: MW p={p_mw:.4f} (r={effect_size:.3f}), "
          f"t p={p_tt:.4f} (d={cohens_d:.3f})")

mw_df = pd.DataFrame(mw_results)
tt_df = pd.DataFrame(tt_results)

# =========================
# STEP 5: save results
# =========================
excel_path = os.path.join(BASE, 'statistical_results.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    mw_df.to_excel(writer, sheet_name='mann_whitney', index=False)
    tt_df.to_excel(writer, sheet_name='t_test', index=False)

    comparison = pd.DataFrame({
        'metric': metrics,
        'mann_whitney_p': mw_df['p_value'].values,
        'mann_whitney_effect_size': mw_df['effect_size'].values,
        'ttest_p': tt_df['p_value'].values,
        'cohens_d': tt_df['cohens_d'].values,
        'normality_healthy': tt_df['normality_healthy'].values,
        'normality_DMD': tt_df['normality_DMD'].values,
        'tests_agree': (mw_df['p_value'].values < 0.05) == (tt_df['p_value'].values < 0.05),
    })
    comparison.to_excel(writer, sheet_name='test_comparison', index=False)

    raw = pd.DataFrame({
        'subject': list(healthy['subject']) + list(dmd['subject']),
        'group':   ['healthy'] * len(healthy) + ['DMD'] * len(dmd),
        'alpha': list(healthy['alpha_mean']) + list(dmd['alpha_mean']),
        'beta':  list(healthy['beta_mean'])  + list(dmd['beta_mean']),
        'D':     list(healthy['D_mean'])     + list(dmd['D_mean']),
        'FA':    list(healthy['FA_mean'])    + list(dmd['FA_mean']),
        'MD':    list(healthy['MD_mean'])    + list(dmd['MD_mean']),
    })
    raw.to_excel(writer, sheet_name='raw_values', index=False)
print(f"\nSaved to {excel_path}")

# =========================
# STEP 6: five-panel comparison plot
# =========================
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Healthy vs DMD comparison (TA muscle)', fontsize=14)
for ax, metric in zip(axes, metrics):
    h_vals = healthy[metric].dropna()
    d_vals = dmd[metric].dropna()
    bp = ax.boxplot([h_vals, d_vals], tick_labels=['Healthy', 'DMD'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.scatter([1] * len(h_vals), h_vals, color='blue', zorder=5, alpha=0.7)
    ax.scatter([2] * len(d_vals), d_vals, color='red', zorder=5, alpha=0.7)
    p = mw_df[mw_df['metric'] == metric]['p_value'].values[0]
    ax.set_title(f'{metric}\np={p:.4f}')
    ax.set_ylabel(metric)
plt.tight_layout()
plot_path = os.path.join(BASE, 'stats_comparison.png')
plt.savefig(plot_path, dpi=150)
print(f"Saved plot to {plot_path}\nDone.")
