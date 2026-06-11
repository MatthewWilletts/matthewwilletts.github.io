"""Generate exact data tables for the figures in neural_operators_lectures.tex.

Everything is closed-form or numerically exact (quadrature against known
kernels, exact spectral steppers) -- computed, not sketched.

Run:  python3 make_data.py   (from inside nop_figs/)
"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.default_rng(0)


def save(name, cols, header):
    np.savetxt(OUT / name, np.column_stack(cols), header=header,
               comments='', fmt='%.8g')


# ----------------------------------------------------------------------
# Fig A: the Green's function of 1D Dirichlet Poisson, and its smoothing
#   -u'' = f on [0,1], u(0)=u(1)=0:  G(x,y) = min(x,y) (1 - max(x,y))
# ----------------------------------------------------------------------
n = 64
xg = (np.arange(n) + 0.5) / n
X, Y = np.meshgrid(xg, xg, indexing='ij')
G = np.minimum(X, Y) * (1 - np.maximum(X, Y))
rows = []
for i in range(n):
    for j in range(n):
        rows.append((xg[i], xg[j], G[i, j]))
np.savetxt(OUT / 'green_heat.dat', np.array(rows), header='x y z',
           comments='', fmt='%.6f')

# a rough f (random Fourier series with slowly decaying spectrum + Dirichlet
# taper) and the smooth u = int G f, computed by fine quadrature
m = 2048
xf = (np.arange(m) + 0.5) / m
f = np.zeros(m)
for k in range(1, 65):
    f += rng.standard_normal() / np.sqrt(k) * np.sin(np.pi * k * xf)
Gf = np.minimum.outer(xf, xf) * (1 - np.maximum.outer(xf, xf))
u = Gf @ f / m
sub = slice(0, m, 4)
save('green_f.dat', [xf[sub], f[sub] / np.abs(f).max()], 'x v')
save('green_u.dat', [xf[sub], u[sub] / np.abs(u).max()], 'x v')
print('green: max|f| %.3f  max|u| %.5f  (smoothing ratio %.1f)'
      % (np.abs(f).max(), np.abs(u).max(), np.abs(f).max() / np.abs(u).max()))

# ----------------------------------------------------------------------
# Fig B: Fourier multipliers of three solution operators, and the cost of
# truncating them at K modes on typical inputs (|f_k| ~ 1/k)
#   Poisson: m(k) = 1/(2 pi k)^2      heat (nu T = 5e-4): exp(-nuT (2pi k)^2)
#   advection: |m(k)| = 1
# ----------------------------------------------------------------------
ks = np.arange(1, 129)
m_poi = 1.0 / (2 * np.pi * ks) ** 2
m_poi = m_poi / m_poi[0]
m_heat = np.exp(-5e-4 * (2 * np.pi * ks) ** 2)
m_adv = np.ones_like(ks, dtype=float)
save('mult.dat', [ks.astype(float), m_poi, m_heat, m_adv], 'k poi heat adv')

a2 = (1.0 / ks) ** 2     # input power spectrum |f_k|^2 ~ k^-2
Ks = np.arange(1, 128)


def trunc_err(m2):
    out = (m2 * a2)
    tot = out.sum()
    return np.sqrt(np.array([out[K:].sum() for K in Ks]) / tot)


save('trunc.dat', [Ks.astype(float), np.maximum(trunc_err(m_poi**2), 1e-16),
                   np.maximum(trunc_err(m_heat**2), 1e-16),
                   trunc_err(m_adv**2)], 'K poi heat adv')

# ----------------------------------------------------------------------
# Fig C: singular values of three discretized solution operators (n = 256)
#   Poisson solve (Dirichlet Green matrix), heat semigroup (periodic,
#   nu T = 5e-4), translation by 0.3 (periodic)
# ----------------------------------------------------------------------
n = 256
xg = (np.arange(n) + 0.5) / n
Gm = np.minimum.outer(xg, xg) * (1 - np.maximum.outer(xg, xg)) / n
s_poi = np.linalg.svd(Gm, compute_uv=False)
s_poi = s_poi / s_poi[0]

kk = np.fft.fftfreq(n, d=1.0 / n)        # integer frequencies
s_heat = np.sort(np.exp(-5e-4 * (2 * np.pi * kk) ** 2))[::-1]
s_adv = np.ones(n)

r = np.arange(1, n + 1, dtype=float)
save('svd.dat', [r, np.maximum(s_poi, 1e-16), np.maximum(s_heat, 1e-16),
                 s_adv], 'r poi heat adv')

# ----------------------------------------------------------------------
# Fig D: quadrature error vs mesh size for a fixed kernel and test function
# (the Nystrom view of mesh transfer).  Periodic domain, equal weights.
#   smooth kernel:  exp(cos(2 pi d) / 0.3)   (analytic -> spectral rate)
#   kinked kernel:  1 - 2 dist(d)            (C^0     -> O(m^-2))
# ----------------------------------------------------------------------
def vfun(y):
    return np.sin(2 * np.pi * y) + 0.3 * np.cos(6 * np.pi * y)


def k_smooth(d):
    return np.exp(np.cos(2 * np.pi * d) / 0.3)


def k_kink(d):
    dist = np.minimum(np.mod(d, 1.0), 1.0 - np.mod(d, 1.0))
    return 1.0 - 2.0 * dist


X0 = 0.37
mref = 1 << 16
yref = (np.arange(mref) + 0.5) / mref
ref_s = np.mean(k_smooth(X0 - yref) * vfun(yref))
ref_k = np.mean(k_kink(X0 - yref) * vfun(yref))
ms, es, ek = [], [], []
for p in range(2, 11):
    mq = 1 << p
    yq = (np.arange(mq) + 0.5) / mq
    es.append(abs(np.mean(k_smooth(X0 - yq) * vfun(yq)) - ref_s))
    ek.append(abs(np.mean(k_kink(X0 - yq) * vfun(yq)) - ref_k))
    ms.append(mq)
save('quad.dat', [np.array(ms, float),
                  np.maximum(np.array(es), 1e-16),
                  np.maximum(np.array(ek), 1e-16)], 'm smooth kink')
print('quad: smooth errors', ['%.1e' % e for e in es[:5]])

# ----------------------------------------------------------------------
# Fig E: rollout of slightly-wrong spectral steppers.  Per-step relative
# multiplier error of 5% x (k/64)^2 (discretization-style: worst at high k).
#   heat stepper: m_k = exp(-nu dt (2pi k)^2), nu = 1e-3, dt = 0.05
#   advection stepper: m_k = exp(-i 2 pi k c dt), c = 0.5 (phase errors)
# ----------------------------------------------------------------------
K = 64
kvec = np.arange(1, K + 1)
u0 = rng.standard_normal(K) / kvec        # |u0_k| ~ 1/k
delta = 0.05 * (kvec / K) ** 2            # relative error, worst at high k

nu, dt, c = 1e-3, 0.05, 0.5
mh = np.exp(-nu * dt * (2 * np.pi * kvec) ** 2)
mh_pert = mh * (1 + delta)
ph = -2 * np.pi * kvec * c * dt
ma = np.exp(1j * ph)
ma_pert = np.exp(1j * ph * (1 + delta))

steps = np.arange(0, 401)
rows_h, rows_a = [], []
for nstep in steps:
    true_h = mh ** nstep * u0
    err_h = np.linalg.norm(mh_pert ** nstep * u0 - true_h) \
        / max(np.linalg.norm(true_h), 1e-300)
    true_a = ma ** nstep * u0
    err_a = np.linalg.norm(ma_pert ** nstep * u0 - true_a) \
        / np.linalg.norm(u0)
    rows_h.append(err_h)
    rows_a.append(err_a)
save('rollout.dat', [steps.astype(float), np.maximum(rows_h, 1e-16),
                     np.maximum(rows_a, 1e-16)], 'n heat adv')
print('rollout: heat err at n=400: %.2e   adv err at n=400: %.2e'
      % (rows_h[-1], rows_a[-1]))

print('done.')
