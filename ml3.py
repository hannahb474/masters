"""
ml3.py -- space-time fractional (Mittag-Leffler) diffusion signal model.

    S(q, Delta) = S0(Delta) * E_alpha( -D * q**beta * Delta**alpha )

E_alpha is the one-parameter Mittag-Leffler function
    E_alpha(z) = sum_k  z**k / Gamma(alpha*k + 1)

with alpha in (0, 1] (fractional time order, sub-diffusion) and beta in (1, 2]
(fractional space order). alpha = 1, beta = 2 gives Gaussian mono-exponential
decay, so departures of alpha/beta from (1, 2) measure departure from free
diffusion.

The non-diffusion-weighted signal is not constant across diffusion time, so the
model uses a SEPARATE baseline amplitude S0(Delta) per diffusion time. Because the
model is linear in these amplitudes they are not optimised but solved in closed
form at each evaluation (variable projection); the optimiser sees only
(theta, alpha, beta). theta = D * q0**beta * Delta0**alpha is the dimensionless
decay at the reference scales and is what is compared across voxels.

E_alpha is evaluated stably: exact exp for alpha = 1, the Taylor series where its
largest term is safe, otherwise the Gorenflo-Mainardi spectral integral by fixed
Gauss-Legendre quadrature.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import gamma as _gamma, erfcx as _erfcx

# tolerances / quadrature setup
CONV_TOL = 1e-12
MAX_TERMS = 200
SERIES_SAFE_LOGMAX = 12.0     # use the series only while log(max term) < this
ALPHA_ONE_TOL = 1e-9          # treat alpha within this of 1 as exactly 1

_N_QUAD = 192
_GL_X, _GL_W = np.polynomial.legendre.leggauss(_N_QUAD)
# map nodes [-1,1] -> w in (0,1) -> u = w/(1-w) in (0, inf)
_W_NODES = 0.5 * (_GL_X + 1.0)
_W_WEIGHTS = 0.5 * _GL_W
_U_NODES = _W_NODES / (1.0 - _W_NODES)
_U_JAC = _W_WEIGHTS / (1.0 - _W_NODES) ** 2


# ---------------------------------------------------------------------------
# Mittag-Leffler function E_alpha(-x), x >= 0
# ---------------------------------------------------------------------------

def _ml_series(x, alpha):
    """Taylor series sum_k (-x)^k / Gamma(alpha k + 1); x 1-D array, x >= 0."""
    s = np.ones_like(x)
    term = np.ones_like(x)
    active = np.ones_like(x, dtype=bool)
    for k in range(1, MAX_TERMS):
        term = term * (-x) / _gamma(alpha * k + 1.0) * _gamma(alpha * (k - 1) + 1.0)
        s[active] += term[active]
        with np.errstate(invalid="ignore", divide="ignore"):
            small = np.abs(term) <= CONV_TOL * np.maximum(np.abs(s), 1e-300)
        active &= ~small
        if not active.any():
            break
    return s


def _ml_spectral(x, alpha):
    """Gorenflo-Mainardi spectral representation, stable for all x >= 0:
        E_alpha(-x) = sin(a pi)/(a pi x) * int_0^inf exp(-u^(1/a))
                      / ((u/x)^2 + 2(u/x)cos(a pi) + 1) du,  a = alpha.
    The integrand is smooth and positive, so fixed-node Gauss-Legendre converges
    quickly. Valid for 0 < alpha < 1 and x >= ~1 (smaller x uses the series)."""
    c = np.cos(alpha * np.pi)
    s = np.sin(alpha * np.pi)
    with np.errstate(over="ignore"):
        expo = np.exp(-np.power(_U_NODES, 1.0 / alpha))
    ratio = _U_NODES[None, :] / x[:, None]
    denom = ratio * ratio + 2.0 * c * ratio + 1.0
    integral = (expo[None, :] / denom) @ _U_JAC
    return s / (alpha * np.pi * x) * integral


def ml3_E_alpha(x, alpha):
    """One-parameter Mittag-Leffler E_alpha(-x) for real x >= 0, 0 < alpha <= 1.
    Vectorised over x."""
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    out = np.ones_like(x)

    if alpha >= 1.0 - ALPHA_ONE_TOL:
        return float(np.exp(-x[0])) if scalar else np.exp(-x)

    nz = x > 0
    if nz.any():
        xn = x[nz]
        with np.errstate(over="ignore"):
            log_max_term = np.power(xn, 1.0 / alpha)      # largest series term ~ exp of this
        use_series = log_max_term < SERIES_SAFE_LOGMAX
        res = np.empty_like(xn)
        if use_series.any():
            res[use_series] = _ml_series(xn[use_series], alpha)
        if (~use_series).any():
            res[~use_series] = _ml_spectral(xn[~use_series], alpha)
        out[nz] = res

    return float(out[0]) if scalar else out


# ---------------------------------------------------------------------------
# signal model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ML3Scales:
    """Fixed protocol reference scales (e.g. median non-zero q and median Delta)
    that make theta dimensionless. Use the SAME values for every voxel and
    subject, otherwise theta is not comparable across fits."""
    q0: float
    Del0: float

    def theta_to_D(self, theta, alpha, beta):
        """Recover raw D (units m^beta / s^alpha)."""
        return theta / (self.q0 ** beta * self.Del0 ** alpha)

    def D_to_theta(self, D, alpha, beta):
        return D * self.q0 ** beta * self.Del0 ** alpha


def ml3_decay(q, Del, theta, alpha, beta, scales):
    """Unit-amplitude decay m = E_alpha( -theta (q/q0)^beta (Delta/Delta0)^alpha )."""
    q = np.asarray(q, dtype=float)
    Del = np.asarray(Del, dtype=float)
    z = theta * (q / scales.q0) ** beta * (Del / scales.Del0) ** alpha
    return ml3_E_alpha(z, alpha)


def ml3_signal(q, Del, theta, alpha, beta, S0, scales):
    """S = S0 * unit-amplitude decay. For a single shared S0 (e.g. simulation)."""
    return S0 * ml3_decay(q, Del, theta, alpha, beta, scales)


def ml3_profile_amplitudes(m, signal, delta, weights=None):
    """Variable-projection baselines: for each diffusion time g,
        S0_g = sum_{i in g} w_i^2 m_i S_i / sum_{i in g} w_i^2 m_i^2   (clipped >= 0).
    Returns the per-shell amplitude (each shell carries its diffusion time's S0)."""
    m = np.asarray(m, float)
    signal = np.asarray(signal, float)
    delta = np.asarray(delta, float)
    w2 = np.asarray(weights, float) ** 2 if weights is not None else np.ones_like(m)
    S0_shell = np.zeros_like(m)
    for g in np.unique(delta):
        sel = delta == g
        den = np.sum(w2[sel] * m[sel] * m[sel])
        S0_shell[sel] = max(np.sum(w2[sel] * m[sel] * signal[sel]) / den, 0.0) if den > 0 else 0.0
    return S0_shell


# ---------------------------------------------------------------------------
# optimiser encoding: x = [log_theta, alpha, beta]   (S0(Delta) profiled out)
# theta is fit in log space (positive, well scaled, non-vanishing gradient);
# alpha/beta are bounded by least_squares.
# ---------------------------------------------------------------------------

ML3_BOUNDS = (
    [np.log(1e-6), 0.05, 1.0],       # lower
    [np.log(1e+6), 1.00, 2.0],       # upper
)
# alpha lower bound is 0.05, not 0: E_alpha is undefined at 0 and the spectral
# decay scale u^(1/alpha) overflows as alpha -> 0.

ML3_X_SCALE = [1.0, 1.0, 1.0]


def ml3_param_encode(x):
    """optimiser vector -> (theta, alpha, beta)."""
    log_theta, alpha, beta = x
    return np.exp(log_theta), alpha, beta


def ml3_x0(theta=1.0, alpha=0.7, beta=1.8):
    """Starting vector in optimiser coordinates (no S0: it is profiled out)."""
    return np.array([np.log(theta), alpha, beta], dtype=float)


# ---------------------------------------------------------------------------
# residuals and fitting
# ---------------------------------------------------------------------------

def ml3_residuals(x, qDels, signal, scales, weights=None):
    """Residual vector for scipy.optimize.least_squares. qDels: (n_shell, 2) of
    [q, Delta]; signal: (n_shell,) shell-averaged signal. The per-diffusion-time
    baselines are solved in closed form (variable projection) before the residual
    is formed; weights (e.g. sqrt(n_avg)) scale both the solve and the residual."""
    theta, alpha, beta = ml3_param_encode(x)
    qDels = np.asarray(qDels, dtype=float)
    signal = np.asarray(signal, dtype=float)
    m = ml3_decay(qDels[:, 0], qDels[:, 1], theta, alpha, beta, scales)
    S0_shell = ml3_profile_amplitudes(m, signal, qDels[:, 1], weights)
    r = S0_shell * m - signal
    if weights is not None:
        r = r * np.asarray(weights, dtype=float)
    return r


def fit_voxel(qDels, signal, scales, weights=None, x0=None, f_scale=None):
    """Fit one voxel; return (theta, alpha, beta, S0, ok). S0 is the mean of the
    profiled per-diffusion-time baselines. f_scale (soft_l1 scale) defaults to 5%
    of the first shell's signal. ok is False if the fit ends on an alpha/beta bound."""
    from scipy.optimize import least_squares

    qDels = np.asarray(qDels, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if x0 is None:
        x0 = ml3_x0()                                     # no S0 to initialise
    if f_scale is None:
        f_scale = max(0.05 * float(signal[0]), 1e-6)

    try:
        res = least_squares(
            ml3_residuals, x0,
            bounds=ML3_BOUNDS, x_scale=ML3_X_SCALE,
            loss="soft_l1", f_scale=f_scale,
            args=(qDels, signal, scales, weights),
            max_nfev=2000,
        )
    except (FloatingPointError, ValueError):
        return (np.nan,) * 4 + (False,)

    theta, alpha, beta = ml3_param_encode(res.x)
    m = ml3_decay(qDels[:, 0], qDels[:, 1], theta, alpha, beta, scales)
    S0_shell = ml3_profile_amplitudes(m, signal, qDels[:, 1], weights)
    delta = qDels[:, 1]
    S0 = float(np.mean([S0_shell[delta == g][0] for g in np.unique(delta)]))

    at_bound = (
        alpha <= ML3_BOUNDS[0][1] + 1e-6 or alpha >= ML3_BOUNDS[1][1] - 1e-6
        or beta <= ML3_BOUNDS[0][2] + 1e-6 or beta >= ML3_BOUNDS[1][2] - 1e-6
    )
    return theta, alpha, beta, S0, bool(res.success) and not at_bound


def averages_to_weights(n_avg):
    """sqrt(n_avg) weights, normalised to mean 1."""
    w = np.sqrt(np.asarray(n_avg, dtype=float))
    return w / w.mean()


def ml3_alpha_half_reference(x):
    """Exact E_{1/2}(-x) = exp(x^2) erfc(x) via erfcx. Validation only."""
    return _erfcx(np.asarray(x, dtype=float))
