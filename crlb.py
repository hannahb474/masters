"""
crlb.py
Cramer-Rao lower bound (CRLB) on DTI-derived MD and FA (report section 5.1).

Signal:  S_i = S0 * exp(-b_i * g_i^T D g_i), fit for the six tensor elements
theta = [Dxx, Dyy, Dzz, Dxy, Dxz, Dyz] with S0 held fixed.

Rician noise: the noise standard deviation is sigma = sqrt(2*sigma_n^2 + A^2 - mu^2),
where sigma_n is the underlying Gaussian (complex-domain) SD, A the true amplitude
(taken as the reference S0) and mu the Rician mean; sigma -> sigma_n at high SNR.

Fisher information over the diffusion-weighted volumes only (b0 excluded):
    F = sum_i (dS/dtheta)_i (dS/dtheta)_i^T / sigma^2,     C = F^-1
    dS/dD_jk = -b_i * S_i * g_j g_k     (factor 2 on the off-diagonal elements).

CRLB(MD) = sqrt(Var(Dxx) + Var(Dyy) + Var(Dzz)) / 3      (diagonal variances)
CRLB(FA)^2 = (dFA/dD)^T C (dFA/dD)                        (full 6x6 covariance)
"""

from __future__ import annotations
import numpy as np
from scipy.special import i0e as _i0e, i1e as _i1e


def tensor_from_params(p):
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = p
    return np.array([[Dxx, Dxy, Dxz], [Dxy, Dyy, Dyz], [Dxz, Dyz, Dzz]], float)


def params_from_tensor(D):
    return np.array([D[0, 0], D[1, 1], D[2, 2], D[0, 1], D[0, 2], D[1, 2]], float)


def md_from_params(p):
    return (p[0] + p[1] + p[2]) / 3.0


def fa_from_params(p):
    w = np.clip(np.linalg.eigvalsh(tensor_from_params(p)), 0, None)
    md = w.mean()
    denom = np.sqrt((w ** 2).sum())
    if denom <= 0:
        return 0.0
    return float(np.sqrt(1.5 * ((w - md) ** 2).sum()) / denom)


def rician_sigma(S0, sigma_n):
    """Rician noise SD at true amplitude A = S0:  sqrt(2 sigma_n^2 + S0^2 - mu^2),
    with mu the Rician mean. Reduces to sigma_n at high SNR."""
    if sigma_n <= 0:
        return 0.0
    a = S0 ** 2 / (2.0 * sigma_n ** 2)
    mu = sigma_n * np.sqrt(np.pi / 2.0) * ((1.0 + a) * _i0e(a / 2.0) + a * _i1e(a / 2.0))
    return float(np.sqrt(max(2.0 * sigma_n ** 2 + S0 ** 2 - mu ** 2, 1e-30)))


def _dS_dD(bvals, bvecs, params, S0):
    """dS/dD (N, 6) with factor 2 on the off-diagonal elements, and signal S (N,)."""
    bvals = np.asarray(bvals, float).ravel()
    bvecs = np.asarray(bvecs, float)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    gx, gy, gz = bvecs
    adc = np.einsum("ji,jk,ki->i", bvecs, tensor_from_params(params), bvecs)
    S = S0 * np.exp(-bvals * adc)
    dS = -bvals[:, None] * S[:, None] * np.column_stack(
        [gx * gx, gy * gy, gz * gz, 2 * gx * gy, 2 * gx * gz, 2 * gy * gz])
    return dS, S


def crlb_covariance(bvals, bvecs, params, S0, sigma, b0_thresh=100.0, ridge=1e-12):
    """6x6 CRLB covariance of the tensor elements. `sigma` is the Gaussian noise SD
    (sigma_n); the Rician correction is applied at A = S0. Baseline volumes
    (b < b0_thresh) are excluded from the Fisher information."""
    bvals = np.asarray(bvals, float).ravel()
    bvecs = np.asarray(bvecs, float)
    if bvecs.shape[0] != 3:
        bvecs = bvecs.T
    dw = bvals >= b0_thresh
    dS, _ = _dS_dD(bvals[dw], bvecs[:, dw], params, S0)

    sig = rician_sigma(S0, sigma)
    F = (dS.T @ dS) / sig ** 2
    F = F + ridge * np.eye(6) * np.trace(F) / 6.0        # mild regularisation
    try:
        return np.linalg.inv(F)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(F)


def fa_gradient(params, eps=1e-6):
    """Numerical d FA / d D (6-vector)."""
    g = np.zeros(6)
    p = np.asarray(params, float)
    for k in range(6):
        step = eps * max(1.0, abs(p[k]))
        dp = np.zeros(6)
        dp[k] = step
        g[k] = (fa_from_params(p + dp) - fa_from_params(p - dp)) / (2 * step)
    return g


def crlb_md_fa(bvals, bvecs, params, S0, sigma, b0_thresh=100.0):
    """Return (var_MD, var_FA), the CRLB variances. MD uses the diagonal element
    variances; FA uses the full 6x6 covariance. Take sqrt for the SD bound."""
    C = crlb_covariance(bvals, bvecs, params, S0, sigma, b0_thresh=b0_thresh)
    var_md = (C[0, 0] + C[1, 1] + C[2, 2]) / 9.0         # (sqrt(sum of diag)/3)^2
    gfa = fa_gradient(params)
    var_fa = float(gfa @ C @ gfa)
    return max(var_md, 0.0), max(var_fa, 0.0)


if __name__ == "__main__":
    # Rician SD reduces to the Gaussian SD at high SNR, inflates at low SNR
    for snr in (2, 5, 10, 40):
        print("SNR %2d: rician_sigma/sigma_n = %.3f" % (snr, rician_sigma(snr, 1.0)))

    # adding measurements must not worsen the CRLB
    D = np.diag([2.2e-3, 1.3e-3, 1.3e-3])
    p = params_from_tensor(D)
    print("MD =", md_from_params(p), " FA =", round(fa_from_params(p), 3))

    dirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                     [1, 1, 0], [1, 0, 1], [0, 1, 1]], float)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    S0, sigma = 1000.0, 25.0

    bvecs1 = np.vstack([[0, 0, 0], dirs]).T
    bvals1 = np.array([0.] + [800.] * 6)
    v_md1, v_fa1 = crlb_md_fa(bvals1, bvecs1, p, S0, sigma)
    print("single-shell b800:  std(MD)=%.3e  std(FA)=%.4f" % (np.sqrt(v_md1), np.sqrt(v_fa1)))

    bvecs2 = np.vstack([[0, 0, 0], dirs, dirs]).T
    bvals2 = np.array([0.] + [800.] * 6 + [400.] * 6)
    v_md2, v_fa2 = crlb_md_fa(bvals2, bvecs2, p, S0, sigma)
    print("two-shell b400+800: std(MD)=%.3e  std(FA)=%.4f" % (np.sqrt(v_md2), np.sqrt(v_fa2)))

    assert v_md2 <= v_md1 + 1e-18 and v_fa2 <= v_fa1 + 1e-18
    print("OK: adding measurements does not worsen the CRLB.")
