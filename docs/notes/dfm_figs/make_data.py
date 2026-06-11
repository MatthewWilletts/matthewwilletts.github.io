"""Generate exact data tables for the TikZ/pgfplots figures in
diffusion_flow_matching_lectures.tex.

Everything here is closed-form or numerically exact (RK4 / Euler-Maruyama on
known scores) -- the figures are computed, not sketched.

Run:  python3 make_data.py   (from inside dfm_figs/)
"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.default_rng(0)


def save(name, cols, header):
    arr = np.column_stack(cols)
    np.savetxt(OUT / name, arr, header=header, comments='', fmt='%.6f')


def save_blocks(name, blocks, header):
    """Blocks separated by nan rows; plot with `unbounded coords=jump`."""
    parts = []
    for b in blocks:
        parts.append(b)
        parts.append(np.full((1, b.shape[1]), np.nan))
    arr = np.vstack(parts[:-1])
    np.savetxt(OUT / name, arr, header=header, comments='', fmt='%.6f')


def quantiles_normal(n):
    z = np.sort(rng.standard_normal(400000))
    levels = (np.arange(n) + 0.5) / n
    return np.quantile(z, levels)


# ----------------------------------------------------------------------
# Fig 1: schedules as curves in the (alpha, sigma) plane
# ----------------------------------------------------------------------
tt = np.linspace(0, 1, 11)  # tick times 0, 0.1, ..., 1

# cosine: quarter circle, uniform arc speed
a = np.cos(np.pi * tt / 2); s = np.sin(np.pi * tt / 2)
save('sched_ticks_cos.dat', [a, s], 'a s')

# VP, DDPM-standard linear beta in [0.1, 20]
integ = 0.1 * tt + 9.95 * tt ** 2
a = np.exp(-0.5 * integ); s = np.sqrt(1 - a ** 2)
save('sched_ticks_vp.dat', [a, s], 'a s')

# linear / rectified
save('sched_ticks_lin.dat', [1 - tt, tt], 'a s')

# VE: alpha = 1, geometric sigma 0.02 -> 2 (leaves the frame)
sig = 0.02 * 100.0 ** tt
save('sched_ticks_ve.dat', [np.ones_like(tt), sig], 'a s')

# ----------------------------------------------------------------------
# Fig 2: score / Tweedie displacement field for a 3-component GMM,
# linear schedule, three noise levels
# ----------------------------------------------------------------------
CENTERS = np.array([[-1.2, -0.4], [0.0, 0.8], [1.2, -0.4]])
S2 = 0.15 ** 2

def gmm_score_displacement(X, t):
    """sigma_t^2 * score(x,t) = alpha*E[x0|x] - x  (Tweedie displacement)."""
    al, sg = 1 - t, t
    V = al ** 2 * S2 + sg ** 2
    mus = al * CENTERS                                     # (3,2)
    d2 = ((X[:, None, :] - mus[None, :, :]) ** 2).sum(-1)  # (N,3)
    logw = -d2 / (2 * V)
    logw -= logw.max(axis=1, keepdims=True)
    r = np.exp(logw); r /= r.sum(axis=1, keepdims=True)
    post_mean = r @ mus                                    # (N,2)
    return post_mean - X, np.sqrt(V), mus

g = np.linspace(-2, 2, 13)
GX, GY = np.meshgrid(g, g)
X = np.column_stack([GX.ravel(), GY.ravel()])

print('--- fig 2 circle metadata (cx cy r per component) ---')
for tag, t in [('t010', 0.10), ('t045', 0.45), ('t090', 0.90)]:
    D, r, mus = gmm_score_displacement(X, t)
    k = 0.30 / np.abs(D).max() * np.sqrt(np.abs(D).max())  # sqrt-compress
    L = np.linalg.norm(D, axis=1, keepdims=True)
    Ds = D / np.maximum(L, 1e-12) * np.sqrt(L) * (0.30 / np.sqrt(L.max()))
    save(f'score_{tag}.dat', [X[:, 0], X[:, 1], Ds[:, 0], Ds[:, 1]],
         'x y u v')
    print(tag, 'r=%.3f' % r, ' centers:',
          ' '.join('(%.3f,%.3f)' % (m[0], m[1]) for m in mus))

# ----------------------------------------------------------------------
# Fig 3: VP diffusion -- density profiles, PF-ODE trajectories, one
# reverse-SDE path.  Data = 0.5 N(-1, 0.1^2) + 0.5 N(+1, 0.1^2).
# ----------------------------------------------------------------------
B0, B1 = 0.1, 19.9
def alpha_vp(t):
    return np.exp(-0.5 * (B0 * t + 0.5 * B1 * t ** 2))
def beta(t):
    return B0 + B1 * t

S2D = 0.1 ** 2
MODES = np.array([-1.0, 1.0])

def vp_score(x, t):
    al = alpha_vp(t); sg2 = 1 - al ** 2
    V = al ** 2 * S2D + sg2
    mus = al * MODES
    d2 = (x[..., None] - mus) ** 2
    logw = -d2 / (2 * V)
    logw -= logw.max(axis=-1, keepdims=True)
    r = np.exp(logw); r /= r.sum(axis=-1, keepdims=True)
    return (r * (-(x[..., None] - mus) / V)).sum(-1)

def vp_density(x, t):
    al = alpha_vp(t); sg2 = 1 - al ** 2
    V = al ** 2 * S2D + sg2
    mus = al * MODES
    return (0.5 / np.sqrt(2 * np.pi * V)
            * np.exp(-(x[..., None] - mus) ** 2 / (2 * V))).sum(-1)

xg = np.linspace(-3.2, 3.2, 321)
for k, tk in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    rho = vp_density(xg, tk)
    w = 0.13 / rho.max()                    # normalised profile width
    save(f'prof3_{k}.dat', [tk + w * rho, xg], 'tt x')

def vp_velocity(x, t):
    return -0.5 * beta(t) * (x + vp_score(x, t))

# PF-ODE, RK4 backwards from t=1
starts = quantiles_normal(12)
N = 500
ts = np.linspace(1.0, 0.0, N + 1)
h = ts[0] - ts[1]
traj = np.empty((N + 1, len(starts)))
traj[0] = starts
x = starts.copy()
for i in range(N):
    t = ts[i]
    k1 = vp_velocity(x, t)
    k2 = vp_velocity(x - 0.5 * h * k1, t - 0.5 * h)
    k3 = vp_velocity(x - 0.5 * h * k2, t - 0.5 * h)
    k4 = vp_velocity(x - h * k3, t - h)
    x = x - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    traj[i + 1] = x
sub = slice(0, N + 1, 5)
blocks = [np.column_stack([ts[sub], traj[sub, j]]) for j in range(len(starts))]
save_blocks('ode3.dat', blocks, 't x')

# one reverse-SDE path (Euler--Maruyama, backwards)
rng_sde = np.random.default_rng(7)
M = 4000
hs = 1.0 / M
x = np.array([0.10])
path = [(1.0, x[0])]
for i in range(M):
    t = 1.0 - i * hs
    b = beta(t)
    drift = -0.5 * b * x - b * vp_score(x, t)
    x = x - hs * drift + np.sqrt(hs * b) * rng_sde.standard_normal(1)
    path.append((t - hs, x[0]))
path = np.array(path)[::5]
save('sde3.dat', [path[:, 0], path[:, 1]], 't x')

# ----------------------------------------------------------------------
# Fig 4: rectified flow.  Data = two deltas at +-1, linear schedule.
# (a) independent coupling: straight conditional lines cross; the
#     marginal PF-ODE trajectories curve.
# (b) reflow coupling (ODE endpoint pairs): straight and non-crossing.
# ----------------------------------------------------------------------
eps_q = quantiles_normal(7)
lines = []
for x0 in (-1.0, 1.0):
    for e in eps_q:
        lines.append(np.array([[0.0, x0], [1.0, e]]))
save_blocks('lines4.dat', lines, 't x')

def rf_velocity(x, t):
    al, sg = 1 - t, t
    m = np.tanh(al * x / sg ** 2)
    return (x - al * m) / sg - m

starts = quantiles_normal(12)
N = 800
ts = np.linspace(0.998, 0.002, N + 1)
h = ts[0] - ts[1]
traj = np.empty((N + 1, len(starts)))
traj[0] = starts
x = starts.copy()
for i in range(N):
    t = ts[i]
    k1 = rf_velocity(x, t)
    k2 = rf_velocity(x - 0.5 * h * k1, t - 0.5 * h)
    k3 = rf_velocity(x - 0.5 * h * k2, t - 0.5 * h)
    k4 = rf_velocity(x - h * k3, t - h)
    x = x - h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    traj[i + 1] = x
sub = slice(0, N + 1, 8)
blocks = [np.column_stack([ts[sub], traj[sub, j]]) for j in range(len(starts))]
save_blocks('ode4.dat', blocks, 't x')

# reflow pairs: straight lines from (1, x1) to (0, ODE endpoint)
reflow = [np.array([[0.0, traj[-1, j]], [1.0, traj[0, j]]])
          for j in range(len(starts))]
save_blocks('reflow4.dat', reflow, 't x')

# ----------------------------------------------------------------------
# Fig 5: the projection lemma in action.  Data = 0.5 N(-1,0.25^2) +
# 0.5 N(+1,0.25^2), linear schedule, t = 0.6 fixed.  Scatter of
# per-sample noise targets eps against x_t, with the closed-form
# conditional mean E[eps | x_t] they average to.
# ----------------------------------------------------------------------
S5 = 0.25
AT, SGT = 0.4, 0.6
n = 320
x0 = rng.choice([-1.0, 1.0], n) + S5 * rng.standard_normal(n)
ep = rng.standard_normal(n)
xt = AT * x0 + SGT * ep
save('lemma_scatter.dat', [xt, ep], 'xt eps')

xg = np.linspace(-2.6, 2.6, 261)
V = AT ** 2 * S5 ** 2 + SGT ** 2
mus = np.array([-1.0, 1.0])
d2 = (xg[:, None] - AT * mus) ** 2
logw = -d2 / (2 * V); logw -= logw.max(axis=1, keepdims=True)
r = np.exp(logw); r /= r.sum(axis=1, keepdims=True)
Eeps = (r * (SGT * (xg[:, None] - AT * mus) / V)).sum(1)
save('lemma_mean.dat', [xg, Eeps], 'xt eps')

# ----------------------------------------------------------------------
# Fig 6: guidance as density tilting.  p = 0.5 N(-1,0.3^2)+0.5 N(1,0.3^2),
# classifier p(c|x) = sigmoid(3x); tilted ~ p * lik^w, w = 1, 3.
# ----------------------------------------------------------------------
xg = np.linspace(-2.4, 2.4, 481)
p = (0.5 / np.sqrt(2 * np.pi * 0.09)
     * (np.exp(-(xg + 1) ** 2 / 0.18) + np.exp(-(xg - 1) ** 2 / 0.18)))
lik = 1 / (1 + np.exp(-3 * xg))
cols = [xg, p, lik]
for w in (1, 3):
    tw = p * lik ** w
    tw /= np.trapezoid(tw, xg)
    cols.append(tw)
save('guide.dat', cols, 'x p lik w1 w3')

# ----------------------------------------------------------------------
# Fig 7: Brownian motion -- sample paths and quadratic variation
# ----------------------------------------------------------------------
rng_bm = np.random.default_rng(11)
paths = []
for k in range(3):
    dWk = rng_bm.standard_normal(2000) / np.sqrt(2000)
    Wk = np.concatenate([[0], np.cumsum(dWk)])
    tk = np.linspace(0, 1, 2001)
    paths.append(np.column_stack([tk[::4], Wk[::4]]))
save_blocks('bm_paths.dat', paths, 't w')

# quadratic variation of ONE path, viewed at three mesh sizes
nfine = 4096
dW = rng_bm.standard_normal(nfine) / np.sqrt(nfine)
W = np.concatenate([[0], np.cumsum(dW)])
for n in (16, 256, 4096):
    inc = np.diff(W[::nfine // n])
    qv = np.concatenate([[0], np.cumsum(inc ** 2)])
    save(f'qv_{n}.dat', [np.linspace(0, 1, n + 1), qv], 't qv')

# ----------------------------------------------------------------------
# Fig 8: the forward VP SDE shredding data (Euler--Maruyama paths,
# conditional mean curves and 2-sigma envelopes, all exact)
# ----------------------------------------------------------------------
rng_ou = np.random.default_rng(21)
M = 2000
ts_f = np.linspace(0, 1, M + 1)
hf = 1.0 / M
paths = []
for x0c in (1.0, 1.0, 1.0, -1.0, -1.0, -1.0):
    x = x0c + 0.1 * rng_ou.standard_normal()
    xs = [x]
    for i in range(M):
        b = beta(ts_f[i])
        x = x - 0.5 * b * x * hf + np.sqrt(b * hf) * rng_ou.standard_normal()
        xs.append(x)
    paths.append(np.column_stack([ts_f[::5], np.array(xs)[::5]]))
save_blocks('ou_paths.dat', paths, 't x')

al = alpha_vp(ts_f); sg = np.sqrt(1 - al ** 2)
for j, mu in enumerate((1.0, -1.0)):
    up, lo = mu * al + 2 * sg, mu * al - 2 * sg
    pt = np.concatenate([ts_f, ts_f[::-1]])[::10]
    px = np.concatenate([up, lo[::-1]])[::10]
    save(f'ou_env_{j}.dat', [pt, px], 't x')
save_blocks('ou_mean.dat',
            [np.column_stack([ts_f[::20], al[::20]]),
             np.column_stack([ts_f[::20], -al[::20]])], 't x')

# ----------------------------------------------------------------------
# Fig 9: solver convergence, error vs NFE, exact velocity fields.
# (a) Euler / Heun / RK4 on the curved rectified-flow PF-ODE
# (b) Euler-in-t vs DDIM (= Euler in (sigma/alpha, x/alpha)) on the VP ODE
# ----------------------------------------------------------------------
def integrate(v, x0s, t0, t1, N, method):
    x = x0s.copy()
    ts = np.linspace(t0, t1, N + 1)
    for i in range(N):
        t, tn = ts[i], ts[i + 1]
        h = tn - t
        k1 = v(x, t)
        if method == 'euler':
            x = x + h * k1
        elif method == 'heun':
            x = x + 0.5 * h * (k1 + v(x + h * k1, tn))
        else:  # rk4
            k2 = v(x + 0.5 * h * k1, t + 0.5 * h)
            k3 = v(x + 0.5 * h * k2, t + 0.5 * h)
            k4 = v(x + h * k3, tn)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x

starts = quantiles_normal(16)
T0, T1 = 0.998, 0.002
ref_rf = integrate(rf_velocity, starts, T0, T1, 4096, 'rk4')
Ns = 2 ** np.arange(0, 11)
print('--- fig 9a: mean |error| at endpoint ---')
for method, cost in (('euler', 1), ('heun', 2), ('rk4', 4)):
    nfes, errs = [], []
    for N in Ns:
        e = np.abs(integrate(rf_velocity, starts, T0, T1, int(N), method)
                   - ref_rf).mean()
        nfes.append(cost * N); errs.append(max(e, 1e-14))
    save(f'conv_{method}.dat', [np.array(nfes), np.array(errs)], 'nfe err')
    print(method, ['%.1e' % e for e in errs[:6]])

# panel (b): sharp modes (s = 0.01) -- the regime where data detail lives
# far below the noise scale and step placement matters
S2B = 1e-4

def vpb_score(x, t):
    al = alpha_vp(t); sg2 = 1 - al ** 2
    V = al ** 2 * S2B + sg2
    mus = al * MODES
    d2 = (x[..., None] - mus) ** 2
    logw = -d2 / (2 * V)
    logw -= logw.max(axis=-1, keepdims=True)
    r = np.exp(logw); r /= r.sum(axis=-1, keepdims=True)
    return (r * (-(x[..., None] - mus) / V)).sum(-1)

def vpb_velocity(x, t):
    return -0.5 * beta(t) * (x + vpb_score(x, t))

def vpb_eps(x, t):
    return -np.sqrt(1 - alpha_vp(t) ** 2) * vpb_score(x, t)

def ddim_grid(x0s, ts):
    x = x0s.copy()
    for i in range(len(ts) - 1):
        t, s = ts[i], ts[i + 1]
        alt, sgt = alpha_vp(t), np.sqrt(1 - alpha_vp(t) ** 2)
        als, sgs = alpha_vp(s), np.sqrt(1 - alpha_vp(s) ** 2)
        e = vpb_eps(x, t)
        x = als * (x - sgt * e) / alt + sgs * e
    return x

# invert log-SNR(t) numerically for the log-SNR-uniform grid
tf = np.linspace(1e-6, 1.0, 200001)
alf = alpha_vp(tf)
lam = np.log(alf ** 2 / (1 - alf ** 2 + 1e-300))   # decreasing in t

def t_of_lam(l):
    return np.interp(-l, -lam, tf)

TMIN = 1e-3
ref_vp = integrate(vpb_velocity, starts, 1.0, 0.0, 8192, 'rk4')
Nb = 2 ** np.arange(1, 9)
print('--- fig 9b: mean |error| at endpoint ---')
rows = {'eulert': [], 'ddimt': [], 'ddimsnr': []}
for N in Nb:
    N = int(N)
    rows['eulert'].append(np.abs(
        integrate(vpb_velocity, starts, 1.0, 0.0, N, 'euler') - ref_vp).mean())
    rows['ddimt'].append(np.abs(
        ddim_grid(starts, np.linspace(1.0, 0.0, N + 1)) - ref_vp).mean())
    lgrid = np.linspace(lam[-1], np.interp(TMIN, tf, lam), N)
    tsg = np.array([t_of_lam(l) for l in lgrid] + [0.0])
    rows['ddimsnr'].append(np.abs(ddim_grid(starts, tsg) - ref_vp).mean())
for name, errs in rows.items():
    save(f'conv_{name}.dat', [Nb.astype(float), np.array(errs)], 'nfe err')
    print(name, ['%.1e' % e for e in errs])

print('done.')
