"""Manuscript projected-gradient ascent: MC10240 x-only control at T=3pi.

Dimensionless H(t)=[8*u(t)*sigma_x+(xi(t)+q*cos(t))*sigma_z]/2.
q=gamma*theta/omega; time=omega*t; initial +x; evaluation at q=0.
uy=0; |u|<=1. Noise sigma/omega=0,0.4,0.8; omega*tau_c=pi.
Noisy terminal objectives at T/pi=3 use 10240 independent OU paths
per training bank, and independent 10240-path evaluation banks. T=0 is exact.
Only the manuscript projected-gradient ascent update is used, with Armijo line search.
The fixed default RNG seeds reproduce the previous UNOPTIMIZED starting pool and
noise samples, permitting a controlled algorithm comparison. No old optimized
waveforms are used as initial controls. All initial controls are newly randomized. No old waveforms/results or
Hermite propagation are present. The zero-noise row uses analytic propagation.
The 12 panels show QFI(t), x-control, rx(t), and d sqrt(F)/d(omega*t).
For x-only control, the manuscript ellipsoid projection reduces exactly to clipping.
The same projection routine also implements general positive weights.

Run with python spin_sensor_mc10240_PGA_T3pi.py.
"""
import os,sys,argparse
from pathlib import Path
PROGRAM_DIRECTORY=Path(__file__).resolve().parent
_early=argparse.ArgumentParser(add_help=False)
_early.add_argument('--output-dir',default=str(PROGRAM_DIRECTORY/'outputs'/'paper_mc10240_PGA_T3pi'))
_known,_rest=_early.parse_known_args()
RUNTIME_OUTPUT=Path(_known.output_dir).resolve()
if '--smoke-test' in _rest:RUNTIME_OUTPUT=RUNTIME_OUTPUT/'smoke_test'
if os.name=='nt':
    _env=Path(sys.executable).resolve().parent
    os.environ['PATH']=os.pathsep.join(str(p) for p in (_env,_env/'Library'/'bin',_env/'Scripts'))+os.pathsep+os.environ.get('PATH','')
for _variable in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_variable]='1'
os.environ['NUMBA_CACHE_DIR']=str(RUNTIME_OUTPUT/'numba_cache')
os.environ['MPLCONFIGDIR']=str(RUNTIME_OUTPUT/'matplotlib_config')
import math,json,hashlib,time,contextlib,multiprocessing as mp
from datetime import datetime
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from scipy.linalg import solve_sylvester,expm
from numba import njit as _numba_njit
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

E_OVER_OMEGA=8.
E=E_OVER_OMEGA
SIGMA_OVER_OMEGA=.8
OMEGA_TAU_C=np.pi
POLAR_PHASE_EPS=1e-12
TRAJECTORY_BATCH=512
SAVE_FULL_EVALUATION_NOISE=False
USE_NUMBA=True
IN_NOTEBOOK=False
TRAINING_SAMPLES=10240
EVALUATION_SAMPLES=10240
CONTROL_BASE_INTERVALS=96
TERMINAL_TIMES_PI=(3.,)
NOISE_STRENGTHS=(0.,.4,.8)
DISPLAY_TIME=3*np.pi
SX=np.array([[0,1],[1,0]],complex)
SY=np.array([[0,-1j],[1j,0]],complex)
SZ=np.diag([1.,-1.]).astype(complex)
PAULI=(SX,SY,SZ)
PLUS=np.array([1.,1.],complex)/np.sqrt(2.)
MINUS=np.array([1.,-1.],complex)/np.sqrt(2.)
COLORS={'optimized':'#1689D4','cpmg':'#E58000','no_control':'#777777'}

NAMES=('no_control','optimized','cpmg')
SIGMAS=NOISE_STRENGTHS
RATE_NAMES=('optimized','cpmg')
QFI_Y_LOWER_LIMITS={'a1':-1.5,'b1':-.8,'c1':-.2}
STYLES={'optimized':'-','cpmg':'--','no_control':'-.'}
LABELS={'optimized':'Optimized','cpmg':'CPMG','no_control':'No control'}
T_FINAL=3*math.pi



def _njit(**kwargs):
    if not USE_NUMBA:
        return lambda function: function
    if IN_NOTEBOOK:
        kwargs['cache'] = False
    return _numba_njit(**kwargs)

@_njit(inline='always', cache=True)
def _nb_coeff(x, y, z, dt):
    q = x * x + y * y + z * z
    a = dt / 2.0
    x2 = a * a * q
    v = math.sqrt(x2)
    c = math.cos(v)
    f = a if v == 0.0 else a * math.sin(v) / v
    if x2 < 0.0001:
        fp = a ** 3 * (-1.0 / 6.0 + x2 / 60.0 - x2 * x2 / 1680.0 + x2 ** 3 / 90720.0)
        fpp = a ** 5 * (1.0 / 60.0 - x2 / 840.0 + x2 * x2 / 30240.0 - x2 ** 3 / 1995840.0)
    else:
        fp = (a * c - f) / (2.0 * q)
        fpp = -a * a * f / (4.0 * q) - 3.0 * fp / (2.0 * q)
    return (c, f, -a * f / 2.0, fp, -a * fp / 2.0, fpp)

@_njit(inline='always', cache=True)
def _nb_h(p, x, y, z):
    return (z * p[0] + (x - 1j * y) * p[1], (x + 1j * y) * p[0] - z * p[1])

@_njit(inline='always', cache=True)
def _nb_pauli(p, axis):
    if axis == 0:
        return (p[1], p[0])
    if axis == 1:
        return (-1j * p[1], 1j * p[0])
    return (p[0], -p[1])

@_njit(inline='always', cache=True)
def _nb_u(p, x, y, z, c, dagger):
    h = _nb_h(p, x, y, z)
    a = (1j if dagger else -1j) * c[1]
    return (c[0] * p[0] + a * h[0], c[0] * p[1] + a * h[1])

@_njit(inline='always', cache=True)
def _nb_du(p, axis, x, y, z, c, dagger):
    h = x if axis == 0 else y if axis == 1 else z
    hp = _nb_h(p, x, y, z)
    pa = _nb_pauli(p, axis)
    sign = 1j if dagger else -1j
    return (2 * h * c[2] * p[0] + sign * (2 * h * c[3] * hp[0] + c[1] * pa[0]), 2 * h * c[2] * p[1] + sign * (2 * h * c[3] * hp[1] + c[1] * pa[1]))

@_njit(inline='always', cache=True)
def _nb_d2(p, axis, x, y, z, c):
    h = x if axis == 0 else y
    hp = _nb_h(p, x, y, z)
    pz = _nb_pauli(p, 2)
    pa = _nb_pauli(p, axis)
    return (4 * h * z * c[4] * p[0] - 1j * (4 * h * z * c[5] * hp[0] + 2 * c[3] * (h * pz[0] + z * pa[0])), 4 * h * z * c[4] * p[1] - 1j * (4 * h * z * c[5] * hp[1] + 2 * c[3] * (h * pz[1] + z * pa[1])))

@_njit(cache=True)
def _nb_forward(noise, kappa, u, dt, indices, keep):
    steps, samples = noise.shape
    states = np.zeros((steps + 1 if keep else 1, samples, 4), np.complex128)
    states[0, :, :2] = 1.0 / math.sqrt(2.0)
    for n in range(steps):
        j = indices[n]
        x, y, k = (u[j, 0], u[j, 1], kappa[n])
        before = n if keep else 0
        after = n + 1 if keep else 0
        for m in range(samples):
            z = noise[n, m]
            c = _nb_coeff(x, y, z, dt[n])
            p = (states[before, m, 0], states[before, m, 1])
            f = (states[before, m, 2], states[before, m, 3])
            pn = _nb_u(p, x, y, z, c, False)
            fn = _nb_u(f, x, y, z, c, False)
            derivative = _nb_du(p, 2, x, y, z, c, False)
            states[after, m, 0], states[after, m, 1] = pn
            states[after, m, 2] = fn[0] + k * derivative[0]
            states[after, m, 3] = fn[1] + k * derivative[1]
    return states

@_njit(cache=True)
def _nb_backward(noise, kappa, u, dt, indices, states, L):
    steps, samples = noise.shape
    gradient = np.zeros_like(u)
    cost = np.zeros((samples, 4), np.complex128)
    for m in range(samples):
        p0, p1 = (states[-1, m, 0], states[-1, m, 1])
        f0, f1 = (states[-1, m, 2], states[-1, m, 3])
        lp0, lp1 = (L[0, 0] * p0 + L[0, 1] * p1, L[1, 0] * p0 + L[1, 1] * p1)
        cost[m, 0] = L[0, 0] * lp0 + L[0, 1] * lp1 - 2 * (L[0, 0] * f0 + L[0, 1] * f1)
        cost[m, 1] = L[1, 0] * lp0 + L[1, 1] * lp1 - 2 * (L[1, 0] * f0 + L[1, 1] * f1)
        cost[m, 2], cost[m, 3] = (-2 * lp0, -2 * lp1)
    for n in range(steps - 1, -1, -1):
        j = indices[n]
        x, y, k = (u[j, 0], u[j, 1], kappa[n])
        gx, gy = (0.0, 0.0)
        for m in range(samples):
            z = noise[n, m]
            c = _nb_coeff(x, y, z, dt[n])
            p = (states[n, m, 0], states[n, m, 1])
            f = (states[n, m, 2], states[n, m, 3])
            cp = (cost[m, 0], cost[m, 1])
            cf = (cost[m, 2], cost[m, 3])
            for axis in range(1):
                dp = _nb_du(p, axis, x, y, z, c, False)
                df = _nb_du(f, axis, x, y, z, c, False)
                mixed = _nb_d2(p, axis, x, y, z, c)
                g = -2 * (np.conj(cp[0]) * dp[0] + np.conj(cp[1]) * dp[1] + np.conj(cf[0]) * (df[0] + k * mixed[0]) + np.conj(cf[1]) * (df[1] + k * mixed[1])).real
                if axis == 0:
                    gx += g
                else:
                    gy += g
            cpn = _nb_u(cp, x, y, z, c, True)
            extra = _nb_du(cf, 2, x, y, z, c, True)
            cfn = _nb_u(cf, x, y, z, c, True)
            cost[m, 0], cost[m, 1] = (cpn[0] + k * extra[0], cpn[1] + k * extra[1])
            cost[m, 2], cost[m, 3] = cfn
        gradient[j, 0] += gx / samples
        gradient[j, 1] += gy / samples
    return gradient

def project(v):
    out = np.zeros_like(v)
    out[:, 0] = np.clip(v[:, 0], -1.0, 1.0)
    return out

def qfi_sld(rho, drho):
    rho = (rho + rho.conj().T) / 2
    drho = (drho + drho.conj().T) / 2
    eig, vec = np.linalg.eigh(rho)
    den = np.maximum(eig, 0)[:, None] + np.maximum(eig, 0)[None, :]
    D = vec.conj().T @ drho @ vec
    L = vec @ np.divide(2 * D, den, out=np.zeros_like(D), where=den > 1e-13) @ vec.conj().T
    return (max(0.0, float(np.trace(rho @ L @ L).real)), L)

def coefficients(h, dt):
    q = sum((a * a for a in h))
    a = dt / 2
    x2 = a * a * q
    x = np.sqrt(x2)
    c, f = (np.cos(x), a * np.sinc(x / np.pi))
    small = x2 < 0.0001
    fp, fpp = (np.empty_like(q), np.empty_like(q))
    z = x2[small]
    fp[small] = a ** 3 * (-1 / 6 + z / 60 - z * z / 1680 + z ** 3 / 90720)
    fpp[small] = a ** 5 * (1 / 60 - z / 840 + z * z / 30240 - z ** 3 / 1995840)
    k = ~small
    fp[k] = (a * c[k] - f[k]) / (2 * q[k])
    fpp[k] = -a * a * f[k] / (4 * q[k]) - 3 * fp[k] / (2 * q[k])
    return (c, f, -a * f / 2, fp, -a * fp / 2, fpp)

def pauli(v, axis):
    if axis == 0:
        return v[:, ::-1]
    if axis == 1:
        return np.column_stack((-1j * v[:, 1], 1j * v[:, 0]))
    return np.column_stack((v[:, 0], -v[:, 1]))

def h_apply(v, h):
    x, y, z = h
    return np.column_stack((z * v[:, 0] + (x - 1j * y) * v[:, 1], (x + 1j * y) * v[:, 0] - z * v[:, 1]))

def U(v, h, c, dagger=False):
    sign = 1j if dagger else -1j
    return c[0][:, None] * v + sign * c[1][:, None] * h_apply(v, h)

def dU(v, axis, h, c, dagger=False):
    _, f, cp, fp, _, _ = c
    hi = h[axis]
    sign = 1j if dagger else -1j
    return (2 * hi * cp)[:, None] * v + sign * ((2 * hi * fp)[:, None] * h_apply(v, h) + f[:, None] * pauli(v, axis))

def d2U(v, axis, h, c):
    _, _, _, fp, cpp, fpp = c
    hi, hz = (h[axis], h[2])
    return (4 * hi * hz * cpp)[:, None] * v - 1j * ((4 * hi * hz * fpp)[:, None] * h_apply(v, h) + 2 * fp[:, None] * (hi[:, None] * pauli(v, 2) + hz[:, None] * pauli(v, axis)))

def atomic_npz(path, **arrays):
    """Publish completed arrays only; interrupted writes cannot corrupt checkpoints."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    publish_file(temporary,path)

def json_write(path, data):
    path=Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    publish_file(tmp,path)


def publish_file(temporary,path):
    # Dropbox/antivirus can briefly hold a file during replacement on Windows.
    for attempt in range(30):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt==29:raise
            time.sleep(min(.25,.03*(attempt+1)))

def time_key(T):
    return hashlib.sha256(float(T).hex().encode()).hexdigest()[:16]

def pi_label(value):
    if value == 0:
        return '0'
    ratio = value / np.pi
    q = Fraction(float(ratio)).limit_denominator(16)
    if abs(float(q) - ratio) < 1e-10:
        prefix = '' if q.numerator == 1 else str(q.numerator)
        return prefix + '\\pi' + ('' if q.denominator == 1 else '/' + str(q.denominator))
    return f'{ratio:.4g}' + '\\pi'

def operator_of(vector):
    return sum((vector[j] * PAULI[j] for j in range(3))) / 2

def projection_columns(eta_norm, overlap, J, F, rate):
    """Complex projection w=<eta,V>/||eta||, V=D_t eta (dimensionless units).

    Returns Re(w), Im(w), J_parallel, phi in [0,2*pi), J_perp, rate error.
    J_parallel=abs(w) <= J; the phase is NOT the unsigned alignment beta.
    """
    if eta_norm <= 1e-140:
        return [np.nan] * 6
    w = complex(overlap) / eta_norm
    parallel = abs(w)
    residual = J * J - parallel * parallel
    if residual < -1e-10 * max(1.0, J * J):
        raise RuntimeError('Complex projection exceeds the full covariant speed.')
    perpendicular = math.sqrt(max(0.0, residual))
    phase = float(np.angle(w) % (2 * np.pi)) if parallel > POLAR_PHASE_EPS * max(1.0, J) else np.nan
    if np.isfinite(phase) and phase >= 2 * np.pi:
        phase = 0.0
    error = abs(rate - 4 * math.sqrt(F) * w.real)
    if error > 1e-07:
        raise RuntimeError('Projected QFI accumulation identity failed.')
    return [w.real, w.imag, parallel, phase, perpendicular, error]

def geometry_point(x, r, s, dotr, dots, u):
    if SIGMA_OVER_OMEGA == 0:
        F = max(0.0, float(np.dot(s, s)))
        rate = 2 * float(np.dot(s, dots))
        J = abs(np.cos(x)) * np.sqrt(max(0.0, 1 - r[2] ** 2)) / 2
        cosine = float(np.clip(rate / (4 * np.sqrt(F) * J), -1, 1)) if F > 1e-24 and J > 1e-12 else np.nan
        beta = float(np.arccos(cosine)) if np.isfinite(cosine) else np.nan
        eta_norm = math.sqrt(F) / 2
        overlap = complex(rate / 8, -np.cos(x) * s[2] / 4)
        return [x, J, cosine, beta, F, rate, 0.0] + projection_columns(eta_norm, overlap, J, F, rate)
    rho = np.eye(2) / 2 + operator_of(r)
    derivative = operator_of(s)
    dotrho, dotderivative = (operator_of(dotr), operator_of(dots))
    H = operator_of(E_OVER_OMEGA * np.array([u[0], u[1], 0.0]))
    L = solve_sylvester(rho, rho, 2 * derivative)
    L = (L + L.conj().T) / 2
    residual = dotrho + 1j * (H @ rho - rho @ H)
    LD = solve_sylvester(rho, rho, 2 * residual)
    LD = (LD + LD.conj().T) / 2
    dotL = solve_sylvester(rho, rho, 2 * dotderivative - dotrho @ L - L @ dotrho)
    dotL = (dotL + dotL.conj().T) / 2
    eig, vec = np.linalg.eigh(rho)
    if eig.min() < -1e-09:
        raise RuntimeError('Nonpositive averaged state in geometry propagation.')
    W = vec * np.sqrt(np.maximum(eig, 0)) @ vec.conj().T
    eta = L @ W / 2
    covariant = (dotL + 1j * (H @ L - L @ H) + L @ LD / 2) @ W / 2
    F = max(0.0, float(np.trace(rho @ L @ L).real))
    J = float(np.linalg.norm(covariant))
    complex_overlap = complex(np.vdot(eta, covariant))
    overlap = complex_overlap.real
    eta_norm = float(np.linalg.norm(eta))
    cosine = float(np.clip(overlap / (eta_norm * J), -1, 1)) if F > 1e-280 and J > 1e-140 else np.nan
    beta = float(np.arccos(cosine)) if np.isfinite(cosine) else np.nan
    rate = float((2 * np.trace(L @ dotderivative) - np.trace(L @ L @ dotrho)).real)
    error = abs(rate - 8 * overlap)
    if error > 1e-07:
        raise RuntimeError('QFI accumulation identity failed.')
    return [x, J, cosine, beta, F, rate, error] + projection_columns(eta_norm, complex_overlap, J, F, rate)

def time_axis(ax, maximum):
    if maximum <= np.pi + 1e-09:
        vals = [0, np.pi / 2, np.pi]
    elif maximum <= 3 * np.pi + 1e-09:
        vals = np.arange(0, maximum + 1e-08, np.pi / 2)
    else:
        vals = np.arange(0, maximum + 1e-08, np.pi)
    vals = np.unique(np.r_[vals, maximum])
    ax.set_xlim(0, maximum)
    ax.set_xticks(vals, ['$' + pi_label(x) + '$' for x in vals])

def ou_samples(times, samples, seed, keep_endpoints=True, sigma=None, correlation=None):
    """Independent OU realizations with exact joint endpoint/integral sampling."""
    sigma = SIGMA_OVER_OMEGA if sigma is None else float(sigma)
    correlation = OMEGA_TAU_C if correlation is None else float(correlation)
    rng = np.random.Generator(np.random.PCG64(seed))
    dt = np.diff(times)
    xi = sigma * rng.standard_normal(samples)
    endpoints = np.empty((len(times), samples)) if keep_endpoints else None
    if keep_endpoints:
        endpoints[0] = xi
    average = np.empty((len(dt), samples))
    for n, h in enumerate(dt):
        if sigma == 0:
            average[n] = 0.0
            xi.fill(0.0)
        else:
            z = h / correlation
            a, b = (np.exp(-z), -np.expm1(-z))
            variance = sigma ** 2 * -np.expm1(-2 * z)
            term = z ** 3 * (2 / 3 - z / 2 + 7 * z * z / 30 - z ** 3 / 12 + 31 * z ** 4 / 1260 - z ** 5 / 160) if z < 0.02 else 2 * z - 2 * b - b * b
            vint = sigma ** 2 * correlation ** 2 * term
            c1 = sigma ** 2 * correlation * b * b / np.sqrt(variance)
            c2 = np.sqrt(max(0.0, vint - c1 * c1))
            z1, z2 = rng.standard_normal((2, samples))
            average[n] = (correlation * b * xi + c1 * z1 + c2 * z2) / h
            xi = a * xi + np.sqrt(variance) * z1
        if keep_endpoints:
            endpoints[n + 1] = xi
    return (np.ascontiguousarray(average), endpoints)

class NoiseBank:

    def __init__(self, times, samples, seed, keep_endpoints=True, *, sigma=None, correlation=None):
        self.times = np.asarray(times)
        self.dt = np.ascontiguousarray(np.diff(times))
        self.kappa = np.ascontiguousarray(np.diff(np.sin(times)) / self.dt)
        self.sigma = float(SIGMA_OVER_OMEGA if sigma is None else sigma)
        self.correlation = float(OMEGA_TAU_C if correlation is None else correlation)
        self.noise, self.endpoints = ou_samples(self.times, samples, seed, keep_endpoints, self.sigma, self.correlation)
        self.samples, self.seed = (int(samples), int(seed))

    def slice(self, first, last):
        part = self.__class__.__new__(self.__class__)
        part.times, part.dt, part.kappa = (self.times, self.dt, self.kappa)
        part.noise = self.noise[:, first:last]
        part.endpoints = None if self.endpoints is None else self.endpoints[:, first:last]
        part.samples, part.seed = (last - first, self.seed)
        part.sigma, part.correlation = (self.sigma, self.correlation)
        return part

    def save(self, path):
        data = {'times': self.times, 'seed': self.seed, 'samples': self.samples, 'generator': 'PCG64', 'numpy_version': np.__version__, 'sigma': self.sigma, 'correlation': self.correlation}
        if SAVE_FULL_EVALUATION_NOISE:
            data['noise'] = self.noise
            data['endpoints'] = self.endpoints
        atomic_npz(path, **data)

    @classmethod
    def load(cls, path, keep_endpoints=True):
        with np.load(path) as data:
            times = data['times'].copy()
            seed = int(data['seed'])
            sigma = float(data['sigma']) if 'sigma' in data else SIGMA_OVER_OMEGA
            correlation = float(data['correlation']) if 'correlation' in data else OMEGA_TAU_C
            if 'noise' not in data:
                return cls(times, int(data['samples']), seed, keep_endpoints, sigma=sigma, correlation=correlation)
            bank = cls.__new__(cls)
            bank.times, bank.seed = (times, seed)
            bank.sigma, bank.correlation = (sigma, correlation)
            bank.noise = data['noise'].copy()
            bank.endpoints = data['endpoints'].copy() if keep_endpoints else None
        bank.samples = bank.noise.shape[1]
        bank.dt = np.ascontiguousarray(np.diff(bank.times))
        bank.kappa = np.ascontiguousarray(np.diff(np.sin(bank.times)) / bank.dt)
        return bank

def interval_indices(bank, edges):
    mid = (bank.times[:-1] + bank.times[1:]) / 2
    return np.ascontiguousarray(np.clip(np.searchsorted(edges, mid, side='right') - 1, 0, len(edges) - 2), dtype=np.int64)

def propagate(bank, v, edges, keep=False):
    index = interval_indices(bank, edges)
    controls = np.ascontiguousarray(v * E_OVER_OMEGA)
    if USE_NUMBA:
        return _nb_forward(bank.noise, bank.kappa, controls, bank.dt, index, keep)
    steps, samples = bank.noise.shape
    states = np.zeros((steps + 1 if keep else 1, samples, 4), complex)
    states[0, :, :2] = 1 / np.sqrt(2)
    for n in range(steps):
        previous = n if keep else 0
        after = n + 1 if keep else 0
        h = (np.full(samples, controls[index[n], 0]), np.full(samples, controls[index[n], 1]), bank.noise[n])
        c = coefficients(h, bank.dt[n])
        psi, phi = (states[previous, :, :2].copy(), states[previous, :, 2:].copy())
        states[after, :, :2] = U(psi, h, c)
        states[after, :, 2:] = U(phi, h, c) + bank.kappa[n] * dU(psi, 2, h, c)
    return states

def estimate_from_states(states):
    psi, phi = (states[:, :2], states[:, 2:])
    rho = psi.T @ psi.conj() / len(psi)
    term = phi.T @ psi.conj() / len(psi)
    F, L = qfi_sld(rho, term + term.conj().T)
    lp = psi @ L.T
    influence = 4 * np.sum(phi.conj() * lp, axis=1).real - np.sum(abs(lp) ** 2, axis=1)
    standard_error = float(np.std(influence, ddof=1) / np.sqrt(len(psi))) if len(psi) > 1 else 0.0
    return (F, standard_error, influence)

def mc_evaluate(bank, v, edges):
    return estimate_from_states(propagate(bank, v, edges)[-1])

def sample_bloch(states):
    p, f = (states[:, :2], states[:, 2:])
    coherence = p[:, 0].conj() * p[:, 1]
    derivative = f[:, 0].conj() * p[:, 1] + p[:, 0].conj() * f[:, 1]
    r = np.column_stack((2 * coherence.real, 2 * coherence.imag, abs(p[:, 0]) ** 2 - abs(p[:, 1]) ** 2))
    s = np.column_stack((2 * derivative.real, 2 * derivative.imag, 2 * (p[:, 0].conj() * f[:, 0] - p[:, 1].conj() * f[:, 1]).real))
    return (r, s)

@_njit(cache=True)
def _nb_geometry_sums(states, endpoints, controls, indices, times):
    count, samples, _ = states.shape
    out = np.zeros((count, 12))
    for n in range(1, count):
        hx, hy = (controls[indices[n - 1], 0], controls[indices[n - 1], 1])
        k = math.cos(times[n])
        for j in range(samples):
            p0, p1 = (states[n, j, 0], states[n, j, 1])
            f0, f1 = (states[n, j, 2], states[n, j, 3])
            a = np.conj(p0) * p1
            b = np.conj(f0) * p1 + np.conj(p0) * f1
            rx, ry, rz = (2 * a.real, 2 * a.imag, (np.conj(p0) * p0 - np.conj(p1) * p1).real)
            sx, sy, sz = (2 * b.real, 2 * b.imag, 2 * (np.conj(p0) * f0 - np.conj(p1) * f1).real)
            hz = endpoints[n, j]
            out[n, 0] += rx
            out[n, 1] += ry
            out[n, 2] += rz
            out[n, 3] += sx
            out[n, 4] += sy
            out[n, 5] += sz
            out[n, 6] += hy * rz - hz * ry
            out[n, 7] += hz * rx - hx * rz
            out[n, 8] += hx * ry - hy * rx
            out[n, 9] += hy * sz - hz * sy - k * ry
            out[n, 10] += hz * sx - hx * sz + k * rx
            out[n, 11] += hx * sy - hy * sx
    return out

def geometry_sums(states, endpoints, controls, indices, times):
    if USE_NUMBA:
        return _nb_geometry_sums(states, endpoints, controls, indices, times)
    count, samples, _ = states.shape
    out = np.zeros((count, 12))
    for n in range(1, count):
        p, f = (states[n, :, :2], states[n, :, 2:])
        a = p[:, 0].conj() * p[:, 1]
        b = f[:, 0].conj() * p[:, 1] + p[:, 0].conj() * f[:, 1]
        r = np.column_stack((2 * a.real, 2 * a.imag, abs(p[:, 0]) ** 2 - abs(p[:, 1]) ** 2))
        s = np.column_stack((2 * b.real, 2 * b.imag, 2 * (p[:, 0].conj() * f[:, 0] - p[:, 1].conj() * f[:, 1]).real))
        hx, hy = controls[indices[n - 1]]
        hz = endpoints[n]
        k = np.cos(times[n])
        dr = np.column_stack((hy * r[:, 2] - hz * r[:, 1], hz * r[:, 0] - hx * r[:, 2], hx * r[:, 1] - hy * r[:, 0]))
        ds = np.column_stack((hy * s[:, 2] - hz * s[:, 1] - k * r[:, 1], hz * s[:, 0] - hx * s[:, 2] + k * r[:, 0], hx * s[:, 1] - hy * s[:, 0]))
        out[n, :3] = r.sum(axis=0)
        out[n, 3:6] = s.sum(axis=0)
        out[n, 6:9] = dr.sum(axis=0)
        out[n, 9:] = ds.sum(axis=0)
    return out

@_njit(cache=True)
def _nb_free_stats(noise, dt, endpoints, have_endpoints):
    count, samples = noise.shape
    phase = np.zeros(samples)
    mean = np.empty(count + 1, np.complex128)
    derivative = np.zeros(count + 1, np.complex128)
    mean[0] = 1.0 + 0j
    for n in range(count):
        real, imag, dreal, dimag = (0.0, 0.0, 0.0, 0.0)
        for j in range(samples):
            phase[j] += noise[n, j] * dt[n]
            co, si = (math.cos(phase[j]), math.sin(phase[j]))
            real += co
            imag -= si
            if have_endpoints:
                hz = endpoints[n + 1, j]
                dreal -= hz * si
                dimag -= hz * co
        mean[n + 1] = (real + 1j * imag) / samples
        derivative[n + 1] = (dreal + 1j * dimag) / samples
    return (mean, derivative)

def free_stats(noise, dt, endpoints, have_endpoints):
    if USE_NUMBA:
        return _nb_free_stats(noise, dt, endpoints, have_endpoints)
    count, samples = noise.shape
    phase = np.zeros(samples)
    mean = np.empty(count + 1, complex)
    derivative = np.zeros(count + 1, complex)
    mean[0] = 1.0
    for n in range(count):
        phase += noise[n] * dt[n]
        phasor = np.exp(-1j * phase)
        mean[n + 1] = phasor.mean()
        if have_endpoints:
            derivative[n + 1] = -1j * np.mean(endpoints[n + 1] * phasor)
    return (mean, derivative)

def fresh_random_controls(T, edges, seed):
    """All guesses are freshly randomized; no reference/previous pulse input."""
    rng = np.random.default_rng(seed)
    midpoint = (edges[:-1] + edges[1:]) / 2
    x = midpoint / T
    controls, labels = ([], [])
    for k in range(12):
        count = 4 + k % 6 * 2
        knots = np.linspace(0.0, 1.0, count)
        values = rng.uniform(-1.0, 1.0, count)
        smooth = np.interp(x, knots, values)
        smooth *= rng.uniform(0.65, 1.0) / max(np.max(abs(smooth)), 1e-15)
        controls.append(smooth)
        labels.append(f'new_random_smooth_{k + 1:02d}')
    for k in range(8):
        smooth = sum((rng.normal() * np.sin(j * np.pi * x + rng.uniform(0.0, 2 * np.pi)) / j for j in range(1, 9)))
        smooth *= rng.uniform(0.6, 1.15) / max(np.max(abs(smooth)), 1e-15)
        controls.append(np.clip(smooth, -1.0, 1.0))
        labels.append(f'new_random_fourier_{k + 1:02d}')
    for k in range(8):
        n = max(2, round(T / np.pi) * (2 + k % 3))
        centers = np.sort(rng.uniform(0.0, T, n))
        widths = rng.uniform(0.5, 1.6, n) * np.pi / E_OVER_OMEGA
        signs = rng.choice(np.array([-1.0, 1.0]), n)
        overlap = np.maximum(0.0, np.minimum(edges[1:, None], centers + widths / 2) - np.maximum(edges[:-1, None], centers - widths / 2))
        pulse = np.clip(overlap @ signs / np.diff(edges), -1.0, 1.0)
        controls.append(pulse)
        labels.append(f'new_random_pulses_{k + 1:02d}')
    return (np.asarray(controls), labels)

def install_input_guard(output):
    """Only this source and this run's output may be read in the workspace."""
    global _INPUT_GUARD_INSTALLED
    if _INPUT_GUARD_INSTALLED:
        return
    output = Path(output).resolve()
    source = Path(__file__).resolve()

    def audit(event, args):
        if event != 'open' or not args or isinstance(args[0], int):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (TypeError, ValueError, OSError):
            return
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 else 0
        reading = 'r' in mode or '+' in mode if isinstance(mode, str) else not flags & os.O_WRONLY
        if not reading or not path.is_relative_to(PROGRAM_DIRECTORY):
            return
        _READ_PATHS.add(str(path))
        if path != source and (not path.is_relative_to(output)):
            _DENIED_PATHS.add(str(path))
            raise PermissionError(f'Fresh MC run forbids previous workspace input: {path}')
    sys.addaudithook(audit)
    _INPUT_GUARD_INSTALLED = True

def save_input_audit(output, name=None):
    json_write(Path(output) / (name or f'input_audit_worker_{os.getpid()}_{time.time_ns()}.json'), {'source': str(Path(__file__).resolve()), 'allowed_data_directory': str(Path(output).resolve()), 'workspace_read_paths': sorted(_READ_PATHS), 'denied_read_attempts': sorted(_DENIED_PATHS), 'previous_waveforms_allowed': False, 'Hermite_propagation_present': False})


def nf_moments(k, h):
    """Return integral exp(i*k*t)dt and integral t*exp(i*k*t)dt on [0,h]."""
    k, h = np.broadcast_arrays(np.asarray(k, float), np.asarray(h, float))
    z = k * h / 2
    sinc = np.sinc(z / np.pi)
    derivative = np.empty_like(z)
    small = abs(z) < 0.0001
    w = z[small]
    derivative[small] = -w / 3 + w ** 3 / 30 - w ** 5 / 840 + w ** 7 / 45360
    w = z[~small]
    derivative[~small] = (w * np.cos(w) - np.sin(w)) / w ** 2
    phase = np.exp(1j * z)
    return (h * phase * sinc, 0.5 * h * h * phase * (sinc - 1j * derivative))

def nf_interval_integral(a, h, amplitude):
    g1, m1 = nf_moments(1 - amplitude, h)
    g2, m2 = nf_moments(-1 - amplitude, h)
    ea = np.exp(1j * np.asarray(a))
    integral = 0.5 * (ea * g1 + ea.conj() * g2)
    derivative = -0.5j * (ea * m1 + ea.conj() * m2)
    return (integral, derivative)

def nf_value_gradient(u, edges):
    u = np.asarray(u, float)
    dt = np.diff(edges)
    amplitude = E * u
    phase = np.r_[0.0, np.cumsum(amplitude * dt)[:-1]]
    local, dlocal = nf_interval_integral(edges[:-1], dt, amplitude)
    factor = np.exp(-1j * phase)
    contributions = factor * local
    I = np.sum(contributions)
    future = np.cumsum(contributions[::-1])[::-1] - contributions
    dI = E * (factor * dlocal - 1j * dt * future)
    return (float(abs(I) ** 2), 2 * np.real(np.conj(I) * dI))

def nf_exact_profile(u, edges, maximum_dt=0.008):
    T = float(edges[-1])
    times = np.sort(np.r_[np.linspace(0, T, math.ceil(T / maximum_dt) + 1), edges, np.arange(0, T + 1e-10, np.pi / 2)])
    times = times[np.r_[True, np.diff(times) > 1e-11]]
    indices = np.clip(np.searchsorted(edges, times, side='left') - 1, 0, len(u) - 1)
    dt = np.diff(edges)
    amplitude = E * u
    phases = np.r_[0.0, np.cumsum(amplitude * dt)]
    full, _ = nf_interval_integral(edges[:-1], dt, amplitude)
    sums = np.r_[0j, np.cumsum(np.exp(-1j * phases[:-1]) * full)]
    h = times - edges[indices]
    local, _ = nf_interval_integral(edges[indices], h, amplitude[indices])
    I = sums[indices] + np.exp(-1j * phases[indices]) * local
    phase = phases[indices] + amplitude[indices] * h
    w = np.exp(1j * phase) * I
    psi = np.exp(-0.5j * phase)[:, None] * PLUS
    derivative = (-0.5j * np.exp(0.5j * phase) * I)[:, None] * MINUS
    qfi = 4 * (np.sum(abs(derivative) ** 2, axis=1) - abs(np.sum(psi.conj() * derivative, axis=1)) ** 2)
    coherence = 2 * abs(psi[:, 0] * psi[:, 1].conj())
    assert np.max(abs(qfi - abs(I) ** 2)) < 1e-10
    return {'times': times, 'control': u[indices], 'phase': phase, 'I': I, 'sensitivity': w, 'psi': psi, 'derivative': derivative, 'qfi': qfi, 'L': coherence}


_READ_PATHS=set()
_DENIED_PATHS=set()
_INPUT_GUARD_INSTALLED=False


@_njit(cache=True)
def reverse_gradient(noise,kappa,controls,dt,indices,final,L):
    """Exact discrete adjoint, reconstructing states by unitarity backwards.

    psi_n = U^dagger psi_(n+1)
    phi_n = U^dagger phi_(n+1) + kappa*(d_z U)^dagger psi_(n+1).
    This avoids saving O(steps*paths) states and the second forward pass.
    It uses all paths and ONE global ensemble SLD, not batchwise QFIs.
    """
    steps,samples=noise.shape
    state=final.copy()
    cost=np.empty_like(final)
    gradient=np.zeros(len(controls))
    for m in range(samples):
        p0,p1=state[m,0],state[m,1]
        f0,f1=state[m,2],state[m,3]
        lp0,lp1=L[0,0]*p0+L[0,1]*p1,L[1,0]*p0+L[1,1]*p1
        cost[m,0]=L[0,0]*lp0+L[0,1]*lp1-2*(L[0,0]*f0+L[0,1]*f1)
        cost[m,1]=L[1,0]*lp0+L[1,1]*lp1-2*(L[1,0]*f0+L[1,1]*f1)
        cost[m,2],cost[m,3]=-2*lp0,-2*lp1
    for n in range(steps-1,-1,-1):
        j=indices[n];x,y,k=controls[j,0],controls[j,1],kappa[n]
        total=0.
        for m in range(samples):
            z=noise[n,m];c=_nb_coeff(x,y,z,dt[n])
            pa=(state[m,0],state[m,1]);fa=(state[m,2],state[m,3])
            pp=_nb_u(pa,x,y,z,c,True)
            ff=_nb_u(fa,x,y,z,c,True)
            extra=_nb_du(pa,2,x,y,z,c,True)
            fp=(ff[0]+k*extra[0],ff[1]+k*extra[1])
            cp=(cost[m,0],cost[m,1]);cf=(cost[m,2],cost[m,3])
            dp=_nb_du(pp,0,x,y,z,c,False)
            df=_nb_du(fp,0,x,y,z,c,False)
            mixed=_nb_d2(pp,0,x,y,z,c)
            total+=-2*(np.conj(cp[0])*dp[0]+np.conj(cp[1])*dp[1]
                +np.conj(cf[0])*(df[0]+k*mixed[0])
                +np.conj(cf[1])*(df[1]+k*mixed[1])).real
            cpn=_nb_u(cp,x,y,z,c,True)
            cfn=_nb_u(cf,x,y,z,c,True)
            ce=_nb_du(cf,2,x,y,z,c,True)
            cost[m,0],cost[m,1]=cpn[0]+k*ce[0],cpn[1]+k*ce[1]
            cost[m,2],cost[m,3]=cfn
            state[m,0],state[m,1]=pp
            state[m,2],state[m,3]=fp
        gradient[j]+=E_OVER_OMEGA*total/samples
    recovery=0.
    for m in range(samples):
        recovery=max(recovery,abs(state[m,0]-1/math.sqrt(2.)),
            abs(state[m,1]-1/math.sqrt(2.)),abs(state[m,2]),abs(state[m,3]))
    return gradient,recovery


def cpmg(T):
    """Analytic signal-matched finite-width pi pulses, benchmark only."""
    centers=np.arange(np.pi/2,T,np.pi)
    width=np.pi/E_OVER_OMEGA
    raw=np.sort(np.r_[0.,T,centers-width/2,centers+width/2])
    raw=np.clip(raw,0.,T)
    edges=raw[np.r_[True,np.diff(raw)>1e-12]]
    mid=(edges[:-1]+edges[1:])/2
    on=np.any(abs(mid[:,None]-centers)<=width/2+1e-12,axis=1)
    return np.column_stack((on.astype(float),np.zeros(len(mid)))),edges


def time_grid(edges,max_dt):
    subdivisions=max(1,int(np.ceil(np.max(np.diff(edges))/max_dt)))
    times=np.concatenate([np.linspace(a,b,subdivisions+1)[:-1] for a,b in zip(edges[:-1],edges[1:])]+[edges[-1:]])
    return times,subdivisions


def coarsen_bank(fine,factor):
    bank=NoiseBank.__new__(NoiseBank)
    bank.times=fine.times[::factor].copy();bank.dt=np.diff(bank.times)
    bank.kappa=np.diff(np.sin(bank.times))/bank.dt
    integrals=(fine.noise*fine.dt[:,None]).reshape(-1,factor,fine.samples).sum(axis=1)
    bank.noise=np.ascontiguousarray(integrals/bank.dt[:,None])
    bank.endpoints=None if fine.endpoints is None else fine.endpoints[::factor].copy()
    bank.samples,bank.seed=fine.samples,fine.seed
    bank.sigma,bank.correlation=fine.sigma,fine.correlation
    return bank


"""Projected gradient ascent with the manuscript's pointwise projection."""
from numba import prange, set_num_threads
set_num_threads(min(2,os.cpu_count() or 1))


@_njit(cache=True,parallel=True)
def pga_forward(noise_paths,kappa,controls,dt,indices):
    samples,steps=noise_paths.shape
    final=np.empty((samples,4),np.complex128)
    for m in prange(samples):
        p=(complex(1/math.sqrt(2.)),complex(1/math.sqrt(2.)))
        f=(0j,0j)
        for n in range(steps):
            j=indices[n];x,y=controls[j,0],controls[j,1];z=noise_paths[m,n]
            c=_nb_coeff(x,y,z,dt[n]);source=_nb_du(p,2,x,y,z,c,False)
            f0=_nb_u(f,x,y,z,c,False);p=_nb_u(p,x,y,z,c,False)
            f=(f0[0]+kappa[n]*source[0],f0[1]+kappa[n]*source[1])
        final[m,0],final[m,1]=p;final[m,2],final[m,3]=f
    return final


@_njit(cache=True,parallel=True)
def pga_reverse(noise_paths,kappa,controls,dt,indices,final,L):
    samples,steps=noise_paths.shape
    chunks=min(samples,16)
    chunk_gradient=np.zeros((chunks,len(controls)))
    recovery=np.zeros(chunks)
    for chunk in prange(chunks):
        for m in range(chunk*samples//chunks,(chunk+1)*samples//chunks):
            p0,p1=final[m,0],final[m,1];f0,f1=final[m,2],final[m,3]
            lp0,lp1=L[0,0]*p0+L[0,1]*p1,L[1,0]*p0+L[1,1]*p1
            cp=(L[0,0]*lp0+L[0,1]*lp1-2*(L[0,0]*f0+L[0,1]*f1),
                L[1,0]*lp0+L[1,1]*lp1-2*(L[1,0]*f0+L[1,1]*f1))
            cf=(-2*lp0,-2*lp1);pa=(p0,p1);fa=(f0,f1)
            for n in range(steps-1,-1,-1):
                j=indices[n];x,y,k=controls[j,0],controls[j,1],kappa[n]
                z=noise_paths[m,n];c=_nb_coeff(x,y,z,dt[n])
                pp=_nb_u(pa,x,y,z,c,True);ff=_nb_u(fa,x,y,z,c,True)
                extra=_nb_du(pa,2,x,y,z,c,True)
                fp=(ff[0]+k*extra[0],ff[1]+k*extra[1])
                dp=_nb_du(pp,0,x,y,z,c,False);df=_nb_du(fp,0,x,y,z,c,False)
                mixed=_nb_d2(pp,0,x,y,z,c)
                chunk_gradient[chunk,j]+=-2*(np.conj(cp[0])*dp[0]+np.conj(cp[1])*dp[1]
                    +np.conj(cf[0])*(df[0]+k*mixed[0])+np.conj(cf[1])*(df[1]+k*mixed[1])).real
                cpn=_nb_u(cp,x,y,z,c,True);cfn=_nb_u(cf,x,y,z,c,True)
                ce=_nb_du(cf,2,x,y,z,c,True)
                cp=(cpn[0]+k*ce[0],cpn[1]+k*ce[1]);cf=cfn;pa=pp;fa=fp
            recovery[chunk]=max(recovery[chunk],abs(pa[0]-1/math.sqrt(2.)),
                abs(pa[1]-1/math.sqrt(2.)),abs(fa[0]),abs(fa[1]))
    return np.sum(chunk_gradient,axis=0)*E_OVER_OMEGA/samples,np.max(recovery)

def ellipsoid_projection(v, q, bound):
    """Euclidean projection: u_alpha=v_alpha/(1+q_alpha*r).

    C is evaluated at the TRIAL control v. All q_alpha must be positive.
    Single-axis and equal-weight cases have analytic roots; otherwise use
    monotone bisection of sum q*v^2/(1+q*r)^2=bound^2 at every time interval.
    """
    v=np.asarray(v,float);q=np.asarray(q,float)
    if v.ndim!=2 or q.shape!=(v.shape[1],) or np.any(q<=0) or bound<=0:
        raise ValueError('Expected (intervals,axes) controls, positive weights and bound.')
    r=np.zeros(len(v));outside=np.sum(q*v*v,axis=1)>bound*bound
    if not np.any(outside):return v.copy(),r
    if len(q)==1 or np.all(q==q[0]):
        r[outside]=(np.sqrt(np.sum(q*v[outside]**2,axis=1))/bound-1.)/q[0]
    else:
        w=v[outside];lo=np.zeros(len(w));hi=np.ones(len(w))
        for _ in range(80):
            bad=np.sum(q*w*w/(1+hi[:,None]*q)**2,axis=1)>bound*bound
            if not np.any(bad):break
            hi[bad]*=2.
        for _ in range(64):
            mid=(lo+hi)/2
            bad=np.sum(q*w*w/(1+mid[:,None]*q)**2,axis=1)>bound*bound
            lo[bad]=mid[bad];hi[~bad]=mid[~bad]
        r[outside]=hi
    u=v/(1+r[:,None]*q)
    return u,r


class Problem:
    """Cache the accepted forward state for the next discrete-adjoint sweep."""
    def __init__(self,edges,bank=None):
        self.edges=np.asarray(edges);self.P=len(edges)-1;self.bank=bank
        self.indices=interval_indices(bank,edges) if bank is not None else None
        if bank is not None and len(np.unique(self.indices))!=self.P:
            raise ValueError('Training time grid must resolve every control interval.')
        if bank is not None:
            if not hasattr(bank,'_pga_path_noise'):
                bank._pga_path_noise=np.ascontiguousarray(bank.noise.T)
            self.path_noise=bank._pga_path_noise
        self.calls=0;self.forward_calls=0;self.last_recovery_error=0.
        self._u=None;self._F=None;self._L=None;self._final=None;self._g=None

    def _evaluate(self,u):
        u=np.ascontiguousarray(u,dtype=float)
        if self._u is not None and np.array_equal(u,self._u):return
        self._u=u.copy();self._g=None
        if self.bank is None:
            self._F,self._g=nf_value_gradient(u,self.edges)
            return
        controls=np.ascontiguousarray(np.column_stack((u,np.zeros(len(u))))*E_OVER_OMEGA)
        self._final=pga_forward(self.path_noise,self.bank.kappa,controls,self.bank.dt,self.indices)
        self.forward_calls+=1
        psi,phi=self._final[:,:2],self._final[:,2:]
        rho=psi.T@psi.conj()/self.bank.samples
        term=phi.T@psi.conj()/self.bank.samples
        self._F,self._L=qfi_sld(rho,term+term.conj().T)

    def value(self,u):
        self._evaluate(u);return float(self._F)

    def value_gradient(self,u):
        self.calls+=1;self._evaluate(u)
        if self._g is None:
            controls=np.ascontiguousarray(np.column_stack((u,np.zeros(len(u))))*E_OVER_OMEGA)
            self._g,error=pga_reverse(self.path_noise,self.bank.kappa,controls,self.bank.dt,
                self.indices,self._final,np.ascontiguousarray(self._L))
            self.last_recovery_error=float(error)
            if error>2e-9:raise RuntimeError(f'Backward state reconstruction lost precision: {error}')
        return float(self._F),self._g.copy()


def pga_checks():
    rng=np.random.default_rng(731921)
    reports=[]
    for q in (np.array([1.]),np.ones(3),np.array([.4,1.3,2.7])):
        v=rng.normal(size=(23,len(q)))*2.1
        u,r=ellipsoid_projection(v,q,1.2)
        slack=np.sum(q*u*u,axis=1)-1.2**2
        equation=float(np.max(abs(u*(1+r[:,None]*q)-v)))
        violation=float(max(0.,np.max(slack)))
        complement=float(np.max(abs(r*slack)))
        assert equation<1e-12 and violation<1e-12 and complement<1e-10
        reports.append({'weights':q.tolist(),'projection_equation_error':equation,
                        'constraint_violation':violation,'complementarity_error':complement})
    edges=np.linspace(0.,.83,12)
    p=Problem(edges);a=rng.uniform(-.8,.8,11)
    F,g=p.value_gradient(a);G=g/(E_OVER_OMEGA*np.diff(edges));eta=.03
    projected,r=ellipsoid_projection((E_OVER_OMEGA*a+eta*G)[:,None],np.ones(1),E_OVER_OMEGA)
    new=projected[:,0]/E_OVER_OMEGA
    assert p.value(new)>F and np.max(abs(new))<=1.
    return {'ellipsoid_projection_KKT_checks':reports,'analytic_test_QFI_increase':p.value(new)-F,
            'physical_gradient_conversion':'G_j=(partial F/partial a_j)/(E*Delta_t_j), a_j=ux_j/E',
            'only_update':'v=ux+eta*G; ux_new=v/(1+r) for qx=1','passed':True}


def solve_control(problem,initial,point,label,limit,tolerance,restarts=1):
    """Pure projected gradient ascent; spectral initial step and Armijo backtracking.

    All accepted steps have the form in the manuscript. The scalar initial step
    is chosen from a Barzilai--Borwein estimate; the actual step is accepted only
    after projection and a monotone Armijo test. No SQP or quasi-Newton direction.
    """
    point=Path(point);started=last=time.perf_counter()
    a=np.clip(np.asarray(initial,float),-1.,1.)
    if point.exists():
        with np.load(point,allow_pickle=False) as d:
            if 'control' in d and 'edges' in d and np.array_equal(d['edges'],problem.edges):
                a=d['control'].copy()
    dt=np.diff(problem.edges);history=[];statuses=[];accepted=0;trials=0
    previous_a=previous_G=None;eta=1.;accepted_eta=0.;last_update=0.;last_r=np.zeros(len(a))
    F,g=problem.value_gradient(a);initialF=F;best=a.copy();bestF=F;bestg=g.copy()
    minimum_gain=0.;max_projection_error=0.;last_trial_v=E_OVER_OMEGA*a
    evaluations=1

    def checkpoint(completed=False):
        pg=projected_gradient(a,g)
        atomic_npz(point,control=a,edges=problem.edges,initial=initial,gradient=g,
                   physical_G=g/(E_OVER_OMEGA*dt),projection_r=last_r,
                   last_accepted_step=accepted_eta,next_initial_step=eta,last_trial_control=last_trial_v)
        report={'label':label,'algorithm':'projected gradient ascent with pointwise ellipsoid projection',
            'initial_QFI':initialF,'QFI':float(F),'projected_gradient':pg,
            'stationarity_tolerance':tolerance,'stationarity_passed':pg<=tolerance,
            'gradient_evaluations':evaluations,'forward_propagations':problem.forward_calls,
            'accepted_updates':accepted,'line_search_trials':trials,'solver_runs':statuses,
            'intervals':len(a),'maximum_dt':float(max(problem.bank.dt)) if problem.bank else None,
            'training_samples':problem.bank.samples if problem.bank else 0,
            'backward_recovery_error':problem.last_recovery_error,'wall_seconds':time.perf_counter()-started,
            'last_projected_update_L2':last_update,'minimum_accepted_QFI_change':minimum_gain,
            'maximum_projection_equation_error':max_projection_error,
            'line_search':'monotone Armijo after projection, c=1e-4, shrink=0.5',
            'step_initialization':'alternating BB1/BB2 scalar estimates on free amplitudes, positive safeguarded',
            'physical_gradient':'G_j=g_j/(E*Delta_t_j), with normalized a_j=ux_j/E',
            'projection_weights':[1.],'uy':0.,'run_completed':completed}
        json_write(point.with_suffix('.json'),report)
        if history:
            np.savetxt(point.with_suffix('.csv'),history,delimiter=',',comments='',
                header='accepted_iteration,QFI,projected_gradient,step_eta,update_L2,line_search_trials,seconds')
        return report

    for restart in range(restarts):
        if projected_gradient(a,g)<=tolerance:break
        if restart:previous_a=previous_G=None;eta=max(1.,eta)
        status='iteration limit'
        for iteration in range(limit):
            pg=projected_gradient(a,g)
            if pg<=tolerance:status='projected stationary point';break
            G=g/(E_OVER_OMEGA*dt)
            if previous_a is not None:
                s=E_OVER_OMEGA*(a-previous_a);y=previous_G-G
                sy=float(np.dot(dt*s,y));ss=float(np.dot(dt*s,s))
                active=((a>=1.-1e-12)&(g>=0.))|((a<=-1.+1e-12)&(g<=0.))
                free=(abs(s)>1e-13)|(~active)
                yy=float(np.dot(dt[free]*y[free],y[free]))
                if sy>1e-25 and ss>0 and yy>0:
                    estimate=ss/sy if accepted%2==0 else sy/yy
                    eta=float(np.clip(estimate,1e-7,1e6))
            step=eta;success=False
            for backtrack in range(45):
                trial_v=E_OVER_OMEGA*a+step*G
                physical,r=ellipsoid_projection(trial_v[:,None],np.ones(1),E_OVER_OMEGA)
                candidate=physical[:,0]/E_OVER_OMEGA;delta=candidate-a
                directional=float(np.dot(g,delta))
                if directional<=0 and np.max(abs(delta))<1e-15:
                    step*=.5;continue
                trialF=problem.value(candidate);trials+=1
                # Near stationarity a fixed objective allowance can overwhelm
                # the Armijo increase and prevent small-gradient convergence.
                # Keep the allowance below one tenth of the requested increase.
                roundoff=min(3e-13*max(1.,abs(F)),1e-5*max(0.,directional))
                if np.isfinite(trialF) and trialF>=F+1e-4*directional-roundoff:
                    success=True;break
                step*=.5
            if not success:status='Armijo line search stalled';break
            oldF=F;previous_a=a.copy();previous_G=G.copy();a=candidate.copy()
            F,g=problem.value_gradient(a);evaluations+=1;accepted+=1;last_r=r.copy()
            last_trial_v=trial_v.copy();accepted_eta=step;eta=min(1.5*step,1e6)
            last_update=float(np.sqrt(np.dot(dt,(E_OVER_OMEGA*delta)**2)))
            minimum_gain=min(minimum_gain,float(F-oldF))
            projection_error=float(np.max(abs(E_OVER_OMEGA*a*(1+r)-trial_v)))
            max_projection_error=max(max_projection_error,projection_error)
            pg=projected_gradient(a,g)
            if F>bestF or (F>=bestF-3e-13*max(1.,abs(bestF)) and pg<projected_gradient(best,bestg)):
                best,bestF,bestg=a.copy(),F,g.copy()
            history.append([accepted,F,pg,step,last_update,backtrack+1,time.perf_counter()-started])
            now=time.perf_counter()
            if accepted==1 or now-last>=20.:
                checkpoint();print(f'{label}: PGA {accepted}, QFI={F:.10f}, projected={pg:.3g}, eta={step:.3g}, {now-started:.1f}s',flush=True);last=now
        statuses.append({'method':'projected gradient ascent','message':status,'accepted_updates':accepted})
        if projected_gradient(a,g)<=tolerance:break
    # Publish the actual final PGA iterate; objective nonmonotonicity is at roundoff only.
    F,g=problem.value_gradient(a);evaluations+=1
    report=checkpoint(True)
    print(f'{label}: PGA DONE QFI={F:.10f}; projected={report["projected_gradient"]:.3g}; passed={report["stationarity_passed"]}',flush=True)
    return a,report



def projected_gradient(u,g):
    return float(np.max(abs(np.clip(u+g,-1.,1.)-u)))


def check_core():
    edges=np.linspace(0.,2.3,8)
    times,_=time_grid(edges,.023)
    bank=NoiseBank(times,600,381119,False,sigma=.8)
    problem=Problem(edges,bank)
    u=np.random.default_rng(1703).uniform(-.8,.8,7)
    F,g=problem.value_gradient(u)
    controls=np.column_stack((u,np.zeros(len(u))))
    states=propagate(bank,controls,edges,True)
    psi,phi=states[-1,:,:2],states[-1,:,2:]
    rho=psi.T@psi.conj()/bank.samples;term=phi.T@psi.conj()/bank.samples
    _,L=qfi_sld(rho,term+term.conj().T)
    reference=_nb_backward(bank.noise,bank.kappa,controls*E_OVER_OMEGA,bank.dt,
        problem.indices,states,np.ascontiguousarray(L))[:,0]*E_OVER_OMEGA
    numerical=[]
    for j in range(len(u)):
        du=np.eye(len(u))[j]*1e-5
        numerical.append((problem.value(u+du)-problem.value(u-du))/2e-5)
    adjoint_error=float(np.max(abs(g-reference)))
    fd_error=float(np.max(abs(g-numerical)))
    assert adjoint_error<2e-9 and fd_error<2e-7,(g,reference,numerical)
    nf=Problem(edges)
    fn,gn=nf.value_gradient(u)
    nfd=np.array([(nf.value(u+np.eye(len(u))[j]*1e-5)-nf.value(u-np.eye(len(u))[j]*1e-5))/2e-5 for j in range(len(u))])
    assert max(abs(gn-nfd))<2e-7
    report={'reverse_vs_stored_state_gradient_error':adjoint_error,
        'reverse_gradient_FD_error':fd_error,'reverse_state_recovery_error':problem.last_recovery_error,
        'noiseless_gradient_FD_error':float(max(abs(gn-nfd)))}
    print('Core checks:',json.dumps(report),flush=True)
    return report


def paper_initials(T,edges,seed):
    controls,labels=fresh_random_controls(T,edges,seed)
    items=list(controls);labels=list(labels)
    rng=np.random.default_rng(seed+991)
    centers=np.arange(np.pi/2,T,np.pi)
    for k in range(8):
        c=centers+rng.normal(0.,(.025+.012*k)*np.pi,len(centers))
        widths=np.pi/E_OVER_OMEGA*rng.uniform(.75,1.25,len(c))
        signs=np.ones(len(c)) if k<4 else rng.choice(np.array([-1.,1.]),len(c))
        overlap=np.maximum(0.,np.minimum(edges[1:,None],c+widths/2)-np.maximum(edges[:-1,None],c-widths/2))
        items.append(np.clip(overlap@signs/np.diff(edges),-1.,1.))
        labels.append(f'new_random_signal_timed_pulses_{k+1:02d}')
    return np.asarray(items),labels


def case_key(sigma,phase):
    return f'sigma_{sigma:.1f}_T_{phase:.0f}pi'


def configure_sigma(sigma):
    global SIGMA_OVER_OMEGA
    SIGMA_OVER_OMEGA=float(sigma)


def case_seed(base,sigma,phase,kind):
    code=f'{base}:{sigma.hex()}:{phase.hex()}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(code).digest()[:8],'little') & ((1<<62)-1)


def load_training(point,coarse=False):
    with np.load(Path(point)/'training_noise.npz') as data:
        times=data['times'].copy();seed=int(data['seed']);samples=int(data['samples'])
        sigma=float(data['sigma']);correlation=float(data['correlation'])
    bank=NoiseBank(times,samples,seed,False,sigma=sigma,correlation=correlation)
    if coarse:bank=coarsen_bank(bank,2)
    return bank


def prepare_case(job):
    sigma,phase,output_name,options=job
    output=Path(output_name);install_input_guard(output);configure_sigma(sigma)
    point=output/case_key(sigma,phase);point.mkdir(exist_ok=True)
    if options['resume'] and (point/'preparation.json').exists():
        return json.loads((point/'preparation.json').read_text(encoding='utf-8'))
    T=phase*np.pi
    P=max(options['base_intervals'],round(options['base_intervals']*phase/2))
    edges=np.linspace(0.,T,P+1)
    seed=case_seed(options['noise_seed'],sigma,phase,'train')
    eval_seed=case_seed(options['noise_seed'],sigma,phase,'evaluation')
    initial_seed=case_seed(options['control_seed'],sigma,phase,'initial')
    bank=None
    if sigma:
        coarse_times,k=time_grid(edges,options['coarse_dt'])
        if k<4:
            coarse_times=np.concatenate([np.linspace(a,b,5)[:-1] for a,b in zip(edges[:-1],edges[1:])]+[edges[-1:]])
        fine_times=np.concatenate([np.linspace(a,b,3)[:-1] for a,b in zip(coarse_times[:-1],coarse_times[1:])]+[coarse_times[-1:]])
        fine=NoiseBank(fine_times,options['training_samples'],seed,False,sigma=sigma)
        fine.save(point/'training_noise.npz')
        bank=coarsen_bank(fine,2)
        del fine
    problem=Problem(edges,bank)
    initials,labels=paper_initials(T,edges,initial_seed)
    scores=np.array([problem.value(u) for u in initials])
    count=options['starts']
    order=list(map(int,np.argsort(-scores)))
    guided=[i for i in order if labels[i].startswith('new_random_signal_timed')]
    broad=[i for i in order if not labels[i].startswith('new_random_signal_timed')]
    if count>=4:
        selected=guided[:2]+broad[:2]
        rest=[i for i in order if i not in selected]
        if count>4:
            rng=np.random.default_rng(initial_seed+291)
            selected.extend(map(int,rng.choice(rest[:max(10,count)],count-4,replace=False)))
    else:selected=order[:count]
    cp,cp_edges=cpmg(T)
    cp_score=problem.value(cp[np.clip(np.searchsorted(cp_edges,(edges[:-1]+edges[1:])/2,side='right')-1,0,len(cp)-1),0])
    atomic_npz(point/'initial_controls.npz',controls=initials,edges=edges,
               labels=np.asarray(labels),scores=scores,selected_indices=np.asarray(selected))
    atomic_npz(point/'analytic_cpmg.npz',control=cp,edges=cp_edges)
    info={'sigma_over_omega':sigma,'T_over_pi':phase,'omega_T':T,
        'control_intervals':P,'initial_seed':initial_seed,'training_seed':seed,'evaluation_seed':eval_seed,
        'initial_sources':labels,'selected_indices':selected,'selected_sources':[labels[i] for i in selected],
        'initial_scores':[float(scores[i]) for i in selected],'CPMG_training_QFI_coarse':cp_score,
        'old_controls_read':False,'CPMG_used_as_initial':False,
        'training_samples':options['training_samples'] if sigma else 0,
        'maximum_coarse_dt':float(max(bank.dt)) if bank else None,
        'maximum_fine_training_dt':float(max(bank.dt))/2 if bank else None}
    json_write(point/'preparation.json',info);save_input_audit(output)
    print(f'{point.name}: {len(initials)} fresh random guesses, {count} starts; samples={info["training_samples"]}.',flush=True)
    return info





def coarse_start(job):
    sigma,phase,start,output_name,options=job
    output=Path(output_name);install_input_guard(output);configure_sigma(sigma)
    point=output/case_key(sigma,phase);destination=point/f'coarse_start_{start:02d}.npz'
    if options['resume'] and destination.with_suffix('.json').exists():
        return json.loads(destination.with_suffix('.json').read_text(encoding='utf-8'))
    with np.load(point/'initial_controls.npz') as data:
        index=int(data['selected_indices'][start]);u=data['controls'][index].copy();edges=data['edges'].copy()
        label=str(data['labels'][index])
    bank=load_training(point,True) if sigma else None
    problem=Problem(edges,bank)
    _,report=solve_control(problem,u,destination,f'{point.name} start {start+1}/{options["starts"]}',
        options['coarse_iterations'],options['coarse_tolerance'],1)
    report.update({'initial_source':label,'initial_index':index,'start':start})
    json_write(destination.with_suffix('.json'),report);save_input_audit(output)
    return report


def finish_case(job):
    set_num_threads(min(4,os.cpu_count() or 1))
    sigma,phase,output_name,options=job
    output=Path(output_name);install_input_guard(output);configure_sigma(sigma)
    point=output/case_key(sigma,phase)
    if options['resume'] and (point/'result.json').exists():
        saved=json.loads((point/'result.json').read_text(encoding='utf-8'))
        if saved['stationarity_passed'] and saved['control_grid_converged']:
            return saved
    info=json.loads((point/'preparation.json').read_text(encoding='utf-8'))
    candidates=[]
    for k in range(options['starts']):
        report=json.loads((point/f'coarse_start_{k:02d}.json').read_text(encoding='utf-8'))
        with np.load(point/f'coarse_start_{k:02d}.npz') as data:u=data['control'].copy();edges=data['edges'].copy()
        candidates.append((report['QFI'],u,edges,k,report))
    candidates.sort(key=lambda item:item[0],reverse=True)
    if sigma==0.:
        # Among numerically equal solutions, use a fixed global-sign convention.
        bestF=candidates[0][0]
        equivalent=[v for v in candidates if bestF-v[0]<1e-9]
        cp,cp_edges=cpmg(phase*np.pi)
        def canonical_score(item):
            u,edges=item[1],item[2]
            reference=cp[np.clip(np.searchsorted(cp_edges,(edges[1:]+edges[:-1])/2,side='right')-1,0,len(cp)-1),0]
            return abs(float(np.dot(u,reference)))
        chosen=max(equivalent,key=canonical_score)
    else:chosen=candidates[0]
    _,u,edges,chosen_start,coarse_report=chosen
    bank=load_training(point,False) if sigma else None
    problem=Problem(edges,bank)
    destination=point/'fine_training.npz'
    if options['resume'] and destination.with_suffix('.json').exists():
        with np.load(destination) as data:u=data['control'].copy()
        fine_report=json.loads(destination.with_suffix('.json').read_text(encoding='utf-8'))
        if not fine_report['stationarity_passed']:
            stamp=time.time_ns()
            for ext in ('.npz','.json','.csv'):
                old=destination.with_suffix(ext)
                if old.exists():old.with_name(old.stem+f'_before_resume_{stamp}'+ext).write_bytes(old.read_bytes())
            u,fine_report=solve_control(problem,u,destination,f'{point.name} continue THIS run fine training',
                options['fine_iterations'],options['final_tolerance'],options['fine_restarts'])
    else:u,fine_report=solve_control(problem,u,destination,f'{point.name} fine training',
        options['fine_iterations'],options['final_tolerance'],options['fine_restarts'])
    grid_history=[fine_report]
    previous_value=fine_report['QFI'];grid_converged=False
    for refinement in range(1,options['grid_refinements']+1):
        edges=np.linspace(0.,phase*np.pi,2*len(u)+1);initial=np.repeat(u,2)
        destination=point/f'control_grid_{refinement}.npz'
        problem=Problem(edges,bank)
        if options['resume'] and destination.with_suffix('.json').exists():
            with np.load(destination) as data:u=data['control'].copy()
            report=json.loads(destination.with_suffix('.json').read_text(encoding='utf-8'))
            if not report['stationarity_passed']:
                stamp=time.time_ns()
                for ext in ('.npz','.json','.csv'):
                    old=destination.with_suffix(ext)
                    if old.exists():old.with_name(old.stem+f'_before_resume_{stamp}'+ext).write_bytes(old.read_bytes())
                u,report=solve_control(problem,u,destination,f'{point.name} continue THIS run grid x{2**refinement}',
                    options['fine_iterations'],options['final_tolerance'],options['fine_restarts'])
        else:u,report=solve_control(problem,initial,destination,f'{point.name} control grid x{2**refinement}',
            options['fine_iterations'],options['final_tolerance'],options['fine_restarts'])
        gain=report['QFI']-previous_value
        if bank:
            _,training_SE,_=mc_evaluate(bank,np.column_stack((u,np.zeros(len(u)))),edges)
            grid_tol=max(options['grid_absolute_tolerance'],.05*training_SE)
        else:training_SE=0.;grid_tol=1e-8
        report.update({'change_from_previous_grid':gain,'control_grid_tolerance':grid_tol,
                       'training_QFI_SE':training_SE})
        json_write(destination.with_suffix('.json'),report)
        grid_history.append(report);previous_value=report['QFI']
        if abs(gain)<grid_tol and report['stationarity_passed']:
            grid_converged=True;break
    final_report=grid_history[-1]
    # Global sign is a physical symmetry here. Fix it only for presentation.
    cp,cp_edges=cpmg(phase*np.pi)
    cp_reference=cp[np.clip(np.searchsorted(cp_edges,(edges[1:]+edges[:-1])/2,side='right')-1,0,len(cp)-1),0]
    sign=1. if float(np.dot(u,cp_reference))>=0. else -1.
    u=sign*u
    v=np.column_stack((u,np.zeros(len(u))))
    atomic_npz(point/'controls.npz',optimized=v,optimized_edges=edges,cpmg=cp,cpmg_edges=cp_edges,
               presentation_global_sign=sign)
    for name,control,bounds in (('optimized',v,edges),('cpmg',cp,cp_edges)):
        np.savetxt(point/f'control_{name}.csv',np.column_stack((bounds[:-1],bounds[1:],E_OVER_OMEGA*control[:,0])),
            delimiter=',',comments='',header='omega_t_start,omega_t_stop,ux_over_omega')
    if sigma:
        finest=max(options['base_intervals'],round(options['base_intervals']*phase/2))*2**options['grid_refinements']
        subdivisions=max(1,math.ceil(phase*np.pi/(finest*options['evaluation_dt'])))
        times=np.linspace(0.,phase*np.pi,finest*subdivisions+1)
        evaluation=NoiseBank(times,options['evaluation_samples'],info['evaluation_seed'],True,sigma=sigma)
        final=propagate(evaluation,v,edges)[-1]
        F,se,influence=estimate_from_states(final)
        jack_var,terminal_var=jackknife_diagnostic(final)
        final_cp=propagate(evaluation,cp,cp_edges)[-1]
        Fcp,secp,influence_cp=estimate_from_states(final_cp)
        jack_cp,terminal_cp=jackknife_diagnostic(final_cp)
        atomic_npz(point/'terminal_bloch_sensitivities.npz',optimized=terminal_var,cpmg=terminal_cp,
                   samples=evaluation.samples)
        Ffree,sefree,_=mc_evaluate(evaluation,np.zeros((1,2)),np.array([0.,phase*np.pi]))
        pair_SE=float(np.std(influence-influence_cp,ddof=1)/np.sqrt(evaluation.samples))
        evaluation.save(point/'evaluation_noise.npz')
    else:
        F=nf_value_gradient(u,edges)[0];Fcp=nf_value_gradient(cp[:,0],cp_edges)[0]
        Ffree=0.;se=secp=sefree=pair_SE=0.;jack_var=jack_cp=None
    assert abs(np.max(abs(u)))<=1.+1e-10
    row={'sigma_over_omega':sigma,'T_over_pi':phase,'omega_T':phase*np.pi,
        'variational':F,'variational_SE':se,'CPMG':Fcp,'CPMG_SE':secp,
        'no_control':Ffree,'no_control_SE':sefree,'paired_variational_minus_CPMG_SE':pair_SE,
        'training_QFI':final_report['QFI'],'final_projected_gradient':final_report['projected_gradient'],
        'stationarity_passed':final_report['stationarity_passed'],'control_grid_converged':grid_converged,
        'control_grid_history':grid_history,'coarse_selected_start':chosen_start,
        'coarse_initial_source':coarse_report['initial_source'],
        'presentation_global_sign':sign,'control_intervals':len(u),
        'training_samples':options['training_samples'] if sigma else 0,
        'evaluation_samples':options['evaluation_samples'] if sigma else 0,
        'jackknife_variational':jack_var,'jackknife_CPMG':jack_cp,
        'training_seed':info['training_seed'],'evaluation_seed':info['evaluation_seed'],
        'old_control_inputs':False,'Hermite_used':False,'global_optimum_proven':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    json_write(point/'result.json',row);save_input_audit(output)
    print(f'{point.name}: EVALUATION F={F:.9f} +/- {se:.4g}; CPMG={Fcp:.9f} +/- {secp:.4g}; '
        f'gradient={row["final_projected_gradient"]:.3g}, grid_pass={grid_converged}',flush=True)
    return row


@_njit(cache=True)
def mean_observables(noise,endpoints,kappa,controls,dt,indices,times):
    """Means, parameter derivatives, time derivatives and delta-method SEs."""
    steps,samples=noise.shape
    state=np.zeros((samples,4),np.complex128);state[:,:2]=1/math.sqrt(2.)
    temporary=np.empty((samples,8))
    data=np.zeros((steps+1,16));data[0,0]=1.
    for n in range(steps):
        hx,hy=controls[indices[n],0],controls[indices[n],1]
        sums=np.zeros(12);rx2=0.
        for m in range(samples):
            c=_nb_coeff(hx,hy,noise[n,m],dt[n])
            pp=(state[m,0],state[m,1]);ff=(state[m,2],state[m,3])
            pn=_nb_u(pp,hx,hy,noise[n,m],c,False)
            fn=_nb_u(ff,hx,hy,noise[n,m],c,False)
            source=_nb_du(pp,2,hx,hy,noise[n,m],c,False)
            f0,f1=fn[0]+kappa[n]*source[0],fn[1]+kappa[n]*source[1]
            p0,p1=pn
            state[m,0],state[m,1]=p0,p1;state[m,2],state[m,3]=f0,f1
            off=np.conj(p0)*p1
            doff=np.conj(f0)*p1+np.conj(p0)*f1
            rx,ry,rz=2*off.real,2*off.imag,(np.conj(p0)*p0-np.conj(p1)*p1).real
            sx,sy,sz=2*doff.real,2*doff.imag,2*(np.conj(p0)*f0-np.conj(p1)*f1).real
            temporary[m,0],temporary[m,1],temporary[m,2]=rx,ry,rz
            temporary[m,3],temporary[m,4],temporary[m,5]=sx,sy,sz
            temporary[m,6]=(np.conj(p0)*p0+np.conj(p1)*p1).real
            temporary[m,7]=2*(np.conj(p0)*f0+np.conj(p1)*f1).real
            hz=endpoints[n+1,m];co=math.cos(times[n+1])
            sums[0]+=rx;sums[1]+=ry;sums[2]+=rz
            sums[3]+=sx;sums[4]+=sy;sums[5]+=sz
            sums[6]+=hy*rz-hz*ry;sums[7]+=hz*rx-hx*rz;sums[8]+=hx*ry-hy*rx
            sums[9]+=hy*sz-hz*sy-co*ry
            sums[10]+=hz*sx-hx*sz+co*rx;sums[11]+=hx*sy-hy*sx
            rx2+=rx*rx
        sums/=samples;data[n+1,:12]=sums
        rx,ry,rz=sums[0],sums[1],sums[2]
        sx,sy,sz=sums[3],sums[4],sums[5]
        den=max(1e-15,1-rx*rx-ry*ry-rz*rz)
        dot=rx*sx+ry*sy+rz*sz
        a=-dot/den;lx,ly,lz=sx-a*rx,sy-a*ry,sz-a*rz
        F=sx*sx+sy*sy+sz*sz+dot*dot/den
        constant=a*a+lx*lx+ly*ly+lz*lz
        meanI=meanI2=0.
        for m in range(samples):
            rdot=lx*temporary[m,0]+ly*temporary[m,1]+lz*temporary[m,2]
            sdot=lx*temporary[m,3]+ly*temporary[m,4]+lz*temporary[m,5]
            influence=2*a*temporary[m,7]+2*sdot-constant*temporary[m,6]-2*a*rdot
            meanI+=influence;meanI2+=influence*influence
        meanI/=samples
        data[n+1,12]=max(0.,F)
        data[n+1,13]=math.sqrt(max(0.,meanI2-samples*meanI*meanI)/(samples-1)/samples)
        data[n+1,14]=math.sqrt(max(0.,rx2-samples*rx*rx)/(samples-1)/samples)
        data[n+1,15]=abs(meanI-F)
    return data


@_njit(cache=True)
def free_observables(noise,endpoints,dt,times):
    steps,samples=noise.shape
    phase=np.zeros(samples)
    data=np.zeros((steps+1,16));data[0,0]=1.
    for n in range(steps):
        c=s=cc=ss=cs=dc=ds=0.
        for m in range(samples):
            phase[m]+=noise[n,m]*dt[n]
            v,w=math.cos(phase[m]),math.sin(phase[m])
            c+=v;s+=w;cc+=v*v;ss+=w*w;cs+=v*w
            dc-=endpoints[n+1,m]*w;ds+=endpoints[n+1,m]*v
        c/=samples;s/=samples;dc/=samples;ds/=samples
        vc=max(0.,cc/samples-c*c);vs=max(0.,ss/samples-s*s);cov=cs/samples-c*s
        sn=math.sin(times[n+1]);co=math.cos(times[n+1])
        if abs(sn)<1e-12:sn=0.
        data[n+1,0]=c;data[n+1,1]=s
        data[n+1,3]=-sn*s;data[n+1,4]=sn*c
        data[n+1,6]=dc;data[n+1,7]=ds
        data[n+1,9]=-co*s-sn*ds;data[n+1,10]=co*c+sn*dc
        data[n+1,12]=sn*sn*(c*c+s*s)
        variance=4*sn**4*(c*c*vc+s*s*vs+2*c*s*cov)
        data[n+1,13]=math.sqrt(max(0.,variance)/(samples-1))
        data[n+1,14]=math.sqrt(vc/(samples-1))
    return data


def noiseless_observables(u,edges,dt):
    profile=nf_exact_profile(u,edges,dt)
    states=np.column_stack((profile['psi'],profile['derivative']))
    r,s=sample_bloch(states)
    data=np.zeros((len(r),16));data[:,:3]=r;data[:,3:6]=s
    times=profile['times']
    h=np.column_stack((E_OVER_OMEGA*profile['control'],np.zeros((len(r),2))))
    data[:,6:9]=np.cross(h,r)
    data[:,9:12]=np.cross(h,s)+np.cos(times)[:,None]*np.cross(np.array([0.,0.,1.]),r)
    data[:,12]=profile['qfi']
    if np.all(u==0.):
        sn=np.sin(times);sn[abs(sn)<1e-12]=0.
        data[:,3:6]=np.column_stack((np.zeros(len(r)),sn,np.zeros(len(r))))
        data[:,12]=sn*sn
    return times,data,profile['control']


def geometry_series(times,data,control):
    result=[[0.,.5,np.nan,np.nan,0.,0.,0.]+[np.nan]*6]
    for j in range(1,len(times)):
        point=geometry_point(float(times[j]),data[j,:3],data[j,3:6],data[j,6:9],data[j,9:12],
            np.array([control[j],0.]))
        if point[4]<1e-20:point[7:]=[np.nan]*6
        result.append(point)
    result=np.asarray(result)
    if np.nanmax(abs(result[:,4]-data[:,12]))>2e-7:
        raise RuntimeError('Profile QFI and SLD geometry QFI disagree.')
    return result


def jackknife_diagnostic(final):
    r,s=sample_bloch(final)
    n=len(r);r0=r.mean(axis=0);s0=s.mean(axis=0)
    def values(rv,sv):
        dot=np.sum(rv*sv,axis=-1)
        return np.sum(sv*sv,axis=-1)+dot*dot/np.maximum(1e-15,1-np.sum(rv*rv,axis=-1))
    value=float(values(r0,s0))
    leave=values((n*r0-r)/(n-1),(n*s0-s)/(n-1))
    mean=float(leave.mean())
    return {'jackknife_SE':float(np.sqrt((n-1)/n*np.sum((leave-mean)**2))),
        'estimated_leading_QFI_bias':(n-1)*(mean-value),'raw_QFI':value},np.column_stack((r,s))


def export_row_profiles(job):
    sigma,output_name,options=job
    output=Path(output_name);configure_sigma(sigma);install_input_guard(output)
    point=output/case_key(sigma,3.)
    with np.load(point/'controls.npz') as d:
        controls={'optimized':(d['optimized'].copy(),d['optimized_edges'].copy()),
                  'cpmg':(d['cpmg'].copy(),d['cpmg_edges'].copy()),
                  'no_control':(np.zeros((1,2)),np.array([0.,DISPLAY_TIME]))}
    arrays={};summary={}
    bank=NoiseBank.load(point/'evaluation_noise.npz') if sigma else None
    for name,(v,edges) in controls.items():
        if sigma:
            times=bank.times;indices=interval_indices(bank,edges)
            data=free_observables(bank.noise,bank.endpoints,bank.dt,times) if name=='no_control' else mean_observables(
                bank.noise,bank.endpoints,bank.kappa,np.ascontiguousarray(v*E_OVER_OMEGA),bank.dt,indices,times)
            control=np.r_[v[0,0],v[indices,0]]
        else:times,data,control=noiseless_observables(v[:,0],edges,options['evaluation_dt'])
        geom=geometry_series(times,data,control)
        table=np.column_stack((times,data,control*E_OVER_OMEGA,geom[:,7:13]))
        np.savetxt(output/f'sigma_{sigma:.1f}_{name}_T3pi_profile.csv',table,delimiter=',',comments='',
            header='omega_t,rx,ry,rz,dq_rx,dq_ry,dq_rz,dt_rx,dt_ry,dt_rz,dt_dq_rx,dt_dq_ry,dt_dq_rz,QFI,QFI_SE,rx_SE,influence_mean_identity_error,ux_over_omega,Re_w,Im_w,J_parallel,phi,J_perp,projection_identity_error')
        arrays[name+'_times']=times;arrays[name+'_data']=data;arrays[name+'_geometry']=geom
        summary[name]={'QFI_end':float(data[-1,12]),'QFI_SE_end':float(data[-1,13]),
            'rx_end':float(data[-1,0]),'maximum_QFI_geometry_difference':float(np.nanmax(abs(geom[:,4]-data[:,12]))),
            'maximum_projection_identity_error':float(np.nanmax(geom[:,12]))}
    arrays['free_curve_times']=arrays['no_control_times'].copy()
    arrays['free_curve_data']=arrays['no_control_data'].copy()
    atomic_npz(output/f'profiles_sigma_{sigma:.1f}.npz',**arrays)
    json_write(output/f'profile_checks_sigma_{sigma:.1f}.json',summary)
    save_input_audit(output)
    print(f'sigma={sigma:g}: QFI/rx/geometry profiles exported.',flush=True)
    return summary


def time_refinement_check(job):
    sigma,phase,output_name,options=job
    output=Path(output_name);install_input_guard(output);configure_sigma(sigma)
    point=output/case_key(sigma,phase)
    if not sigma:return {'sigma':sigma,'phase':phase,'analytic_propagation':True,'passed':True}
    row=json.loads((point/'result.json').read_text(encoding='utf-8'))
    with np.load(point/'controls.npz') as data:v,edges=data['optimized'].copy(),data['optimized_edges'].copy()
    with np.load(point/'evaluation_noise.npz') as data:times=data['times'].copy()
    fine_times=np.concatenate([np.linspace(a,b,5)[:-1] for a,b in zip(times[:-1],times[1:])]+[times[-1:]])
    bank=NoiseBank(fine_times,options['diagnostic_samples'],row['evaluation_seed']+7183,False,sigma=sigma)
    values=[]
    for factor in (4,2,1):
        sub=coarsen_bank(bank,factor) if factor>1 else bank
        F,se,_=mc_evaluate(sub,v,edges)
        values.append({'maximum_dt':float(max(sub.dt)),'QFI':F})
    delta=abs(values[0]['QFI']-values[-1]['QFI'])
    tolerance=max(2e-4,.02*row['variational_SE'])
    report={'sigma':sigma,'phase':phase,'samples':bank.samples,
        'same_fine_OU_paths_all_steps':True,'checks':values,'coarse_fine_QFI_change':delta,
        'comparison_tolerance':tolerance,'passed':delta<tolerance}
    json_write(point/'time_step_check.json',report);save_input_audit(output)
    print(f'{point.name}: coupled-path time refinement change={delta:.3g}, passed={report["passed"]}',flush=True)
    return report


def polar_panel(ax,geometries):
    maximum=max(.5,max(float(np.nanmax(a[:,9])) for a in geometries.values()))
    maximum=math.ceil((maximum-1e-10)*8)/8
    theta=np.linspace(0.,2*np.pi,721)
    radii=np.linspace(maximum/4,maximum,4)
    for radius in radii:
        ax.plot(radius*np.cos(theta),radius*np.sin(theta),color='.84',lw=.45,zorder=0)
    angles=np.arange(8)*np.pi/4
    labels=(r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$',r'$5\pi/4$',r'$3\pi/2$',r'$7\pi/4$')
    for angle,label in zip(angles,labels):
        ax.plot([0,maximum*np.cos(angle)],[0,maximum*np.sin(angle)],color='.86',lw=.4,zorder=0)
        ax.text((maximum+.065)*np.cos(angle),(maximum+.065)*np.sin(angle),label,
            ha='center',va='center',fontsize=7)
    for radius in radii:
        ax.text(radius*np.cos(np.pi/8),radius*np.sin(np.pi/8),f'{radius:.2g}',
            color='.45',fontsize=5.5,ha='center',va='center',
            bbox={'facecolor':'white','edgecolor':'none','pad':.15})
    for name,ls,lw in (('no_control','-.',.75),('cpmg','--',.95),('optimized','-',1.)):
        g=geometries[name]
        ax.plot(g[:,7],g[:,8],color=COLORS[name],ls=ls,lw=lw,zorder=3 if name=='optimized' else 2)
        if np.all(np.isfinite(g[-1,7:9])):ax.plot(*g[-1,7:9],'o',ms=2.6,color=COLORS[name],zorder=4)
    ax.plot(.5,0.,'o',mfc='white',mec='.35',ms=3.4,mew=.6,zorder=5)
    ax.set_aspect('equal',adjustable='box');ax.axis('off')
    ax.set_xlim(-maximum-.10,maximum+.10);ax.set_ylim(-maximum-.10,maximum+.10)


def configure_style(plt):
    plt.rcParams.update({
        'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8,
        'axes.labelsize': 9, 'axes.titlesize': 9, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'axes.linewidth': .65, 'xtick.direction': 'in',
        'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    })


def time_axis(ax, np, maximum, half_pi=False):
    if half_pi:
        ax.set_xticks(np.arange(7) * np.pi / 2,
            [r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$',
             r'$2\pi$', r'$5\pi/2$', r'$3\pi$'])
    else:
        count = round(maximum / np.pi)
        labels = [r'$0$', r'$\pi$'] + [rf'${k}\pi$' for k in range(2, count + 1)]
        ax.set_xticks(np.arange(count + 1) * np.pi, labels)
        ax.set_xticks(np.arange(.5, count, 1.) * np.pi, minor=True)
    ax.set_xlim(0., maximum)


def pale_color(name, matplotlib):
    return tuple(.90 + .10 * v for v in matplotlib.colors.to_rgb(COLORS[name]))


def accumulation_curve(profile, name, np):
    t = profile[name + '_times']
    data = profile[name + '_data']
    geometry = profile[name + '_geometry']
    speed = 2 * geometry[:, 7].copy()
    assert t[0] == 0. and data[0, 12] == 0.
    # Read the saved root-QFI rate directly, avoiding finite differences.
    # The +x probe at theta=0 has the initial right-hand limit +1.
    speed[data[:, 12] == 0.] = np.nan
    speed[0] = 1.
    valid = np.isfinite(speed)
    assert np.all(valid[data[:, 12] > 0.])
    reconstructed = 2 * np.sqrt(data[valid, 12]) * speed[valid]
    error = float(np.max(abs(reconstructed - geometry[valid, 5])))
    assert error < 2e-7, 'Root-QFI accumulation rate does not match saved QFI derivative.'
    return t, speed, {'protocol': name, 'pointwise_QFI_rate_identity_error': error,
        'undefined_rate_times': t[~valid].tolist(), 'initial_rate_is_right_hand_limit': True,
        'initial_sqrt_QFI_rate': float(speed[0]),
        'minimum_sqrt_QFI_rate': float(np.min(speed[valid])),
        'maximum_sqrt_QFI_rate': float(np.max(speed[valid]))}


def accumulation_panel(ax, profile, sigma, output, np):
    curves = {name: accumulation_curve(profile, name, np) for name in RATE_NAMES}
    t, v, unused = curves['optimized']
    ax.fill_between(t, 0., v, where=v >= 0., interpolate=True, color='#D9EAF8', lw=0, zorder=0)
    ax.fill_between(t, 0., v, where=v < 0., interpolate=True, color='#F3D4D0', lw=0, zorder=0)
    checks = []
    for name in RATE_NAMES:
        t, v, check = curves[name]
        ax.plot(t, v, color=COLORS[name], ls=STYLES[name], lw=.95, zorder=3)
        check['sigma_over_omega'] = sigma
        checks.append(check)
        np.savetxt(output / f'sqrt_QFI_accumulation_rate_sigma_{sigma:.1f}_{name}_T3pi.csv',
            np.column_stack((t, v)), delimiter=',', comments='',
            header='omega_t,d_sqrt_Ftheta_d_omega_t')
    ax.axhline(0., color='.5', lw=.55)
    minimum = min(float(np.nanmin(curves[name][1])) for name in RATE_NAMES)
    maximum = max(float(np.nanmax(curves[name][1])) for name in RATE_NAMES)
    padding = .08 * max(maximum - minimum, 1e-6)
    ax.set_ylim(minimum - padding, maximum + padding)
    ax.set_xlabel(r'$\omega t$')
    ax.set_ylabel(r'$\mathrm{d}\sqrt{F_{\theta}}/\mathrm{d}(\omega t)$')
    time_axis(ax, np, T_FINAL)
    return checks


def save_figure(fig, output, stem):
    files = []
    for suffix in ('png', 'pdf', 'svg', 'eps'):
        path = output / f'{stem}.{suffix}'
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=.04)
        files.append(path.name)
    return files


def plot_time_evolution(profiles, controls, E, output, np, matplotlib, plt, Line2D):
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 6.4))
    fig.subplots_adjust(left=.065, right=.985, bottom=.070, top=.885, wspace=.38, hspace=.49)
    for ax, title in zip(axes[0], ('QFI', 'Control field', 'Bloch component', 'Accumulation rate')):
        ax.set_title(title, pad=10)
    rate_checks = []
    for row, sigma in enumerate(SIGMAS):
        profile = profiles[sigma]
        ax = axes[row, 0]
        highest = 0.
        for name in NAMES:
            t, data = profile[name + '_times'], profile[name + '_data']
            # Saved mean-state QFI and its saved Monte Carlo SE: no recomputation.
            F, se = data[:, 12], data[:, 13]
            if sigma:
                ax.fill_between(t, F - se, F + se, color=pale_color(name, matplotlib), lw=0, zorder=1)
            ax.plot(t, F, color=COLORS[name], ls=STYLES[name], lw=.95, zorder=3)
            highest = max(highest, float(np.max(F + se)))
        ax.text(.04, .93, r'$\sigma_\xi/\omega=' + f'{sigma:.1f}' + r'$',
                transform=ax.transAxes, va='top', fontsize=8)
        # Comparable visible margins are needed because the three QFI ranges differ.
        ax.set_ylim(QFI_Y_LOWER_LIMITS[f'{chr(97 + row)}1'], highest * 1.18)
        # Keep a margin below zero while displaying only nonnegative ticks.
        ax.set_yticks([tick for tick in ax.get_yticks() if 0. <= tick <= ax.get_ylim()[1]])
        time_axis(ax, np, T_FINAL, half_pi=True)
        ax.set_xlabel(r'$\omega t$')
        ax.set_ylabel(r'$F_\theta$')

        ax = axes[row, 1]
        control = controls[sigma]
        for name in ('optimized', 'cpmg'):
            ax.stairs(E * control[name][:, 0], control[name + '_edges'], baseline=None,
                      color=COLORS[name], ls=STYLES[name], lw=.95)
        for bound in (-E, E):
            ax.axhline(bound, color='.65', ls=':', lw=.55)
        ax.set_ylim(-9., 9.)
        ax.set_yticks([-8, -4, 0, 4, 8])
        ax.set_ylabel(r'$u_x/\omega$')
        ax.set_xlabel(r'$\omega t$')
        time_axis(ax, np, T_FINAL)

        ax = axes[row, 2]
        for name in NAMES:
            t, data = profile[name + '_times'], profile[name + '_data']
            if sigma:
                ax.fill_between(t, data[:, 0] - data[:, 14], data[:, 0] + data[:, 14],
                                color=pale_color(name, matplotlib), lw=0, zorder=1)
            ax.plot(t, data[:, 0], color=COLORS[name], ls=STYLES[name], lw=.95, zorder=3)
        ax.set_ylim(-.035, 1.045)
        ax.set_yticks([0., .5, 1.])
        ax.set_ylabel(r'$r_x$')
        ax.set_xlabel(r'$\omega t$')
        time_axis(ax, np, T_FINAL)
        rate_checks.extend(accumulation_panel(axes[row, 3], profile, sigma, output, np))
        if row in (1, 2):
            # A smaller pad moves these long vertical labels 3 pt to the right.
            axes[row, 3].yaxis.labelpad = 1.

        for column in range(4):
            axes[row, column].text(-.11, 1.055, f'({chr(97 + row)}{column + 1})',
                                  transform=axes[row, column].transAxes, fontsize=9)
    handles = [Line2D([], [], color=COLORS[n], ls=STYLES[n], lw=1.1, label=LABELS[n]) for n in NAMES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995), ncol=3,
               frameon=False, fontsize=9, handlelength=2.6, columnspacing=2.5)
    files = save_figure(fig, output, 'spin_mc10240_PGA_T3pi_time_evolution')
    plt.close(fig)
    return files, rate_checks


def plot_paper(output,rows):
    profiles={};controls={}
    for sigma in NOISE_STRENGTHS:
        with np.load(Path(output)/f'profiles_sigma_{sigma:.1f}.npz') as d:
            profiles[sigma]={key:d[key].copy() for key in d.files}
        with np.load(Path(output)/case_key(sigma,3.)/'controls.npz') as d:
            controls[sigma]={key:d[key].copy() for key in d.files}
    configure_style(plt)
    files,checks=plot_time_evolution(profiles,controls,E_OVER_OMEGA,Path(output),np,matplotlib,plt,Line2D)
    json_write(Path(output)/'figure_rate_checks.json',checks)
    return files



def worker_init(output):
    install_input_guard(output)


def run_jobs(pool,function,jobs,label):
    if pool is None:return [function(j) for j in jobs]
    pending={pool.submit(function,j) for j in jobs};result=[];last=time.perf_counter()
    while pending:
        done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
        for future in done:result.append(future.result())
        now=time.perf_counter()
        if done or now-last>25.:
            print(f'{label}: {len(result)}/{len(jobs)} completed.',flush=True);last=now
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,parents=[_early])
    parser.add_argument('--check-environment',action='store_true')
    parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
    parser.add_argument('--seed',type=int,default=1508932354608790621)
    parser.add_argument('--control-seed',type=int,default=143805154684888460)
    parser.add_argument('--starts',type=int,default=6)
    parser.add_argument('--coarse-iterations',type=int,default=250)
    parser.add_argument('--fine-iterations',type=int,default=2500)
    parser.add_argument('--fine-restarts',type=int,default=3)
    parser.add_argument('--grid-refinements',type=int,default=3)
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    print('Python executable:',sys.executable,flush=True);print('Python version:',sys.version,flush=True)
    print('Working directory:',Path.cwd(),flush=True)
    if args.check_environment:
        print(json.dumps({'numpy':np.__version__,'noise_strengths':NOISE_STRENGTHS,
            'training_samples':TRAINING_SAMPLES,'evaluation_samples':EVALUATION_SAMPLES,
            'Hermite_used':False,'old_controls_used':False},indent=2));return 0
    output=RUNTIME_OUTPUT;output.mkdir(parents=True,exist_ok=True);install_input_guard(output)
    started=time.perf_counter()
    manifest_path=output/'run_manifest.json'
    if args.resume:
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        noise_seed=manifest['options']['noise_seed'];control_seed=manifest['options']['control_seed']
        current_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if current_hash!=manifest['source_sha256']:
            initial_hash=manifest.get('initial_source_sha256',manifest['source_sha256'])
            revisions=manifest.get('source_revisions',[{'source_sha256':initial_hash,
                'stage':'initial execution','snapshot':'program_initial_optimizer_snapshot.py'}])
            revisions.append({'source_sha256':current_hash,'stage':'resume',
                'recorded_at':datetime.now().astimezone().isoformat(),
                'change':'PGA-only optimizer revision; retained same initial controls and training noise.',
                'snapshot':'program_final_optimizer_snapshot.py'})
            (output/'program_final_optimizer_snapshot.py').write_bytes(Path(__file__).read_bytes())
            manifest.update({'initial_source_sha256':initial_hash,'source_sha256':current_hash,'source_revisions':revisions})
            json_write(manifest_path,manifest)
    else:
        if manifest_path.exists():raise RuntimeError('Use a new --output-dir or explicitly --resume this run.')
        noise_seed=args.seed if args.seed is not None else int.from_bytes(os.urandom(8),'little') & ((1<<61)-1)
        control_seed=args.control_seed
    options={'noise_seed':noise_seed,'control_seed':control_seed,'resume':args.resume,
        'starts':args.starts,'training_samples':TRAINING_SAMPLES,'evaluation_samples':EVALUATION_SAMPLES,
        'base_intervals':CONTROL_BASE_INTERVALS,'coarse_dt':.02,'evaluation_dt':.004,
        'coarse_iterations':args.coarse_iterations,'coarse_tolerance':2e-4,
        'fine_iterations':args.fine_iterations,'fine_restarts':args.fine_restarts,
        'final_tolerance':3e-6,'grid_refinements':args.grid_refinements,
        'grid_absolute_tolerance':5e-4,'diagnostic_samples':256,'smoke':args.smoke_test}
    if args.smoke_test:
        options.update({'training_samples':32,'evaluation_samples':64,'base_intervals':12,
            'starts':2,'coarse_iterations':3,'fine_iterations':3,'fine_restarts':1,
            'grid_refinements':1,'diagnostic_samples':32});args.workers=1
    if args.resume:
        for key in ('starts','training_samples','evaluation_samples','base_intervals','coarse_dt',
                    'evaluation_dt','coarse_iterations','coarse_tolerance','final_tolerance',
                    'grid_absolute_tolerance','diagnostic_samples','smoke'):
            options[key]=manifest['options'][key]
        json_write(output/f'resume_options_{time.time_ns()}.json',options)
    if not args.resume:
        json_write(manifest_path,{'started_at':datetime.now().astimezone().isoformat(),
            'python':sys.executable,'python_version':sys.version,'cwd':str(Path.cwd()),'options':options,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'physics':{'E_over_omega':8.,'omega_tau_c':float(np.pi),'noise_strengths':[0.,.4,.8],
                       'initial_state':'+x','theta':0.,'uy':0.,'abs_ux_bound_over_omega':8.},
            'QFI_points_over_pi':[0.,3.], 'optimizer':'projected gradient ascent; no SLSQP calls','other_columns_terminal_over_pi':3.,
            'old_control_inputs':False,'Hermite_used':False,'command':sys.argv})
    print('Fresh noise seed:',noise_seed,'; fresh control seed:',control_seed,flush=True)
    print('Noisy objectives use ALL',options['training_samples'],'training paths; evaluation uses independent',options['evaluation_samples'],'paths.',flush=True)
    check=check_core();check['PGA_projection']=pga_checks();json_write(output/'core_checks.json',check)
    jobs=[(sigma,phase,str(output),options) for sigma in NOISE_STRENGTHS for phase in TERMINAL_TIMES_PI]
    with contextlib.ExitStack() as stack:
        pool=None
        if args.workers>1:
            pool=stack.enter_context(ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                initializer=worker_init,initargs=(str(output),)))
        run_jobs(pool,prepare_case,jobs,'Fresh initialization')
        start_jobs=[(sigma,phase,k,str(output),options) for phase in reversed(TERMINAL_TIMES_PI)
                    for sigma in (.8,.4,0.) for k in range(options['starts'])]
        run_jobs(pool,coarse_start,start_jobs,'Independent multistart search')
        rows=run_jobs(pool,finish_case,jobs,'Fine propagation, stationarity and control-grid checks')
        rows.sort(key=lambda r:(r['sigma_over_omega'],r['T_over_pi']))
        json_write(output/'results.json',rows)
        refinements=run_jobs(pool,time_refinement_check,jobs,'Coupled-path propagation checks')
        run_jobs(pool,export_row_profiles,[(s,str(output),options) for s in NOISE_STRENGTHS],'Time profiles and geometry')
    table=[[sigma]+[0.]*10 for sigma in NOISE_STRENGTHS]
    for row in rows:
        table.append([row[k] for k in ('sigma_over_omega','omega_T','variational','variational_SE','CPMG','CPMG_SE',
            'no_control','no_control_SE','paired_variational_minus_CPMG_SE','final_projected_gradient','control_intervals')])
    table.sort(key=lambda a:(a[0],a[1]))
    np.savetxt(output/'terminal_QFI_and_errors.csv',table,delimiter=',',comments='',
        header='sigma_over_omega,omega_T,variational_QFI,variational_SE,CPMG_QFI,CPMG_SE,no_control_QFI,no_control_SE,paired_difference_SE,projected_gradient,control_intervals')
    plot_paper(output,rows)
    verification={'all_stationarity_checks_passed':all(r['stationarity_passed'] for r in rows),
        'all_control_grid_checks_passed':all(r['control_grid_converged'] for r in rows),
        'all_time_checks_passed':all(r['passed'] for r in refinements),
        'time_step_checks':refinements,'training_samples_per_noisy_case':options['training_samples'],
        'evaluation_samples_per_noisy_case':options['evaluation_samples'],
        'noisy_terminal_cases':2,'noiseless_cases_use_analytic_propagation':True,
        'all_control_initials_new_random':True,'old_controls_read':False,'Hermite_used':False,
        'QFI_definition':'QFI of mean density matrix; not mean trajectory QFI',
        'error_bars':'one delta-method standard error; independent evaluation paths',
        'global_optimum_proven':False,'sensitivity_plot':'d sqrt(F_theta)/d(omega*t), optimized and CPMG only','optimizer':'projected gradient ascent'}
    verification['passed']=all(verification[k] for k in ('all_stationarity_checks_passed','all_control_grid_checks_passed','all_time_checks_passed'))
    json_write(output/'verification.json',verification);save_input_audit(output,'input_audit_main.json')
    elapsed=time.perf_counter()-started
    json_write(output/'timing_last_run.json',{'total_wall_seconds':elapsed,'workers':args.workers,
        'training_samples':options['training_samples'],'evaluation_samples':options['evaluation_samples'],
        'noise_seed':noise_seed,'control_seed':control_seed,'output_directory':str(output)})
    print(f'Completed in {elapsed:.3f}s. Verification passed: {verification["passed"]}. Output: {output}',flush=True)
    if not verification['passed'] and not args.smoke_test:
        raise RuntimeError('Some convergence checks require further refinement; inspect verification.json and resume this run.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())

