"""OU-Hermite QFI at integer pi to 20pi, adaptive bounded x-only pulses.

Stationary OU: sigma/omega=.8, omega*tau_c=pi. H/omega=[8*u*sigma_x+
(xi/omega+q*cos(omega*t))*sigma_z]/2, q=gamma*theta/omega; theta=0;
initial +x, uy=0, |u|<=1. No Monte Carlo trajectories are generated.
Control amplitudes and positive segment durations are jointly optimized.
Both amplitude and duration gradients exactly differentiate the discrete RK4 map.
Substep counts are held fixed locally to make duration optimization smooth.
Final QFI uses converged Hermite levels and exact piecewise matrix exponentials.
The default seed directory contains the selected starting controls from the
previous Hermite run.
The parity-reduced dynamics are algebraically identical, with half as many state
components. RK4 and its discrete adjoint use preallocated work arrays.
Only even-pi terminal times are optimized. No fast evaluation-budget cutoff is used.
The local convergence criterion is a relative box-projected gradient <=3e-6,
using the starting QFI magnitude as a fixed objective scale. Absolute gradients
are also recorded; control-mesh, Hermite and time checks are separate.
No prior Monte Carlo or earlier 4pi Hermite data are read. --fresh ignores seeds.
Local multistart results do not prove a global optimum.

Run with python spin_sensor_hermite20_even_completed.py.
Fresh independent run: --fresh --output-dir outputs/hermite20_adaptive_run2
"""
import os,sys,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
_early=argparse.ArgumentParser(add_help=False)
_early.add_argument('--output-dir',type=Path,default=ROOT/'outputs'/'hermite_qfi_20pi_even_final')
_early.add_argument('--seed-dir',type=Path,default=ROOT/'data'/'initial_controls')
_known,_rest=_early.parse_known_args()
RUNTIME_OUTPUT=_known.output_dir.resolve()
if '--smoke-test' in _rest:RUNTIME_OUTPUT=RUNTIME_OUTPUT/'smoke_test'
if os.name=='nt':
    _env=Path(sys.executable).resolve().parent
    os.environ['PATH']=os.pathsep.join(str(p) for p in (_env,_env/'Library'/'bin',_env/'Scripts'))+os.pathsep+os.environ.get('PATH','')
for _name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_name]='1'
os.environ['NUMBA_CACHE_DIR']=str(RUNTIME_OUTPUT/'numba_cache')
os.environ['MPLCONFIGDIR']=str(RUNTIME_OUTPUT/'matplotlib_config')
import ast,math,json,time,hashlib,secrets,shutil,multiprocessing as mp
from datetime import datetime
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from scipy.optimize import minimize
from scipy.sparse import bmat,diags,eye,kron,csr_matrix
from scipy.sparse.linalg import expm_multiply
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
E=8.;SIGMA=.8;TAU=np.pi;OPT_MODES=32;OPT_DT=.008
SX=np.array([[0.,1.],[1.,0.]],complex)
SY=np.array([[0.,-1j],[1j,0.]],complex)
SZ=np.diag([1.,-1.]).astype(complex)
PAULI=(SX,SY,SZ)
MODE_COUNTS=(16,24,32,48,64,96,128)
COLORS={'optimized':'#1689D4','cpmg':'#E58000','no_control':'#777777'}


def cross(v):
    x, y, z = v
    return csr_matrix([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


@lru_cache(maxsize=24)
def _hierarchy_generators(levels, amplitude, sigma, correlation):
    size = 3*levels
    I, B = eye(size, format="csr"), eye(levels, format="csr")
    Z = kron(B, cross([0, 0, 1]), format="csr")
    q = np.sqrt(np.arange(1, levels))
    J = diags([q, q], [-1, 1], shape=(levels, levels))
    A0 = kron(diags(-np.arange(levels)/correlation), eye(3), format="csr")
    A0 += sigma*kron(J, cross([0, 0, 1]), format="csr")
    G0 = bmat([[A0, Z, None], [None, A0, -I], [None, I, A0]], format="csr")
    X = kron(B, cross([amplitude, 0, 0]), format="csr")
    Y = kron(B, cross([0, amplitude, 0]), format="csr")
    Gx = kron(eye(3, format="csr"), X, format="csr")
    Gy = kron(eye(3, format="csr"), Y, format="csr")
    return G0, Gx, Gy, float(G0.diagonal().sum())


def operator_of(vector):
    return sum(vector[j]*PAULI[j] for j in range(3))/2


def qfi_sld(rho, drho):
    rho = (rho+rho.conj().T)/2
    drho = (drho+drho.conj().T)/2
    eig, vec = np.linalg.eigh(rho)
    den = np.maximum(eig, 0)[:, None]+np.maximum(eig, 0)[None, :]
    D = vec.conj().T@drho@vec
    L = vec@np.divide(2*D, den, out=np.zeros_like(D), where=den>1e-13)@vec.conj().T
    return max(0., float(np.trace(rho@L@L).real)), L


"""Parity-reduced OU hierarchy and allocation-free RK4/discrete adjoint.

For even n, (x,y,z)=(r_x,s_y,s_z); for odd n, (x,y,z)=(s_x,r_y,r_z).
All omitted components vanish exactly for x-only control at theta=0.
"""

@njit(cache=True, inline='always')
def reduced_rhs(y, ux, co, decay, coupling, out, transpose=False):
    modes = len(y)
    hx = E*ux
    for n in range(modes):
        cx = cy = 0.
        if n:
            cx += coupling[n-1]*y[n-1, 0]
            cy += coupling[n-1]*y[n-1, 1]
        if n+1 < modes:
            cx += coupling[n]*y[n+1, 0]
            cy += coupling[n]*y[n+1, 1]
        x, v, z = y[n, 0], y[n, 1], y[n, 2]
        d = decay[n]
        if transpose:
            out[n, 0] = -d*x+cy+(co*v if n%2 == 0 else 0.)
            out[n, 1] = -d*v-cx+hx*z-(co*x if n%2 else 0.)
            out[n, 2] = -d*z-hx*v
        else:
            out[n, 0] = -d*x-cy-(co*v if n%2 else 0.)
            out[n, 1] = -d*v+cx-hx*z+(co*x if n%2 == 0 else 0.)
            out[n, 2] = -d*z+hx*v


@njit(cache=True)
def control_inner(adjoint, y):
    value = 0.
    for n in range(len(y)):
        value += -adjoint[n, 1]*y[n, 2]+adjoint[n, 2]*y[n, 1]
    return E*value


@njit(cache=True)
def rk4_forward(u, dt, indices, cosine, decay, coupling, keep):
    steps, modes = len(dt), len(decay)
    y = np.zeros((modes, 3))
    y[0, 0] = 1.
    states = np.empty((steps+1 if keep else 1, modes, 3))
    states[0] = y
    k1, k2, k3, k4, temp = [np.empty_like(y) for _ in range(5)]
    for j in range(steps):
        h, ux = dt[j], u[indices[j]]
        reduced_rhs(y, ux, cosine[j, 0], decay, coupling, k1)
        for n in range(modes):
            for c in range(3):
                temp[n,c] = y[n,c]+h/2*k1[n,c]
        reduced_rhs(temp, ux, cosine[j, 1], decay, coupling, k2)
        for n in range(modes):
            for c in range(3):
                temp[n,c] = y[n,c]+h/2*k2[n,c]
        reduced_rhs(temp, ux, cosine[j, 1], decay, coupling, k3)
        for n in range(modes):
            for c in range(3):
                temp[n,c] = y[n,c]+h*k3[n,c]
        reduced_rhs(temp, ux, cosine[j, 2], decay, coupling, k4)
        for n in range(modes):
            for c in range(3):
                y[n,c] += h/6*(k1[n,c]+2*k2[n,c]+2*k3[n,c]+k4[n,c])
        states[j+1 if keep else 0] = y
    return states


@njit(cache=True)
def boundary_adjoint(u, dt, indices, cosine, decay, coupling, states):
    modes = len(decay)
    gradient = np.zeros(len(u))
    lam = np.zeros((modes, 3))
    lam[0, 1:3] = 2*states[-1, 0, 1:3]
    boundaries = np.zeros((len(u)+1, modes, 3))
    boundaries[-1] = lam
    k1, k2, k3, y2, y3, y4, b1, b2, b3, b4, by, z = [np.empty_like(lam) for _ in range(12)]
    for j in range(len(dt)-1, -1, -1):
        h, index = dt[j], indices[j]
        ux, y = u[index], states[j]
        c1, c2, c4 = cosine[j, 0], cosine[j, 1], cosine[j, 2]
        reduced_rhs(y, ux, c1, decay, coupling, k1)
        for n in range(modes):
            for c in range(3):
                y2[n,c] = y[n,c]+h/2*k1[n,c]
        reduced_rhs(y2, ux, c2, decay, coupling, k2)
        for n in range(modes):
            for c in range(3):
                y3[n,c] = y[n,c]+h/2*k2[n,c]
        reduced_rhs(y3, ux, c2, decay, coupling, k3)
        for n in range(modes):
            for c in range(3):
                y4[n,c] = y[n,c]+h*k3[n,c]
                b1[n,c] = h/6*lam[n,c]
                b2[n,c] = h/3*lam[n,c]
                b3[n,c] = h/3*lam[n,c]
                b4[n,c] = h/6*lam[n,c]
                by[n,c] = lam[n,c]
        gradient[index] += control_inner(b4, y4)
        reduced_rhs(b4, ux, c4, decay, coupling, z, True)
        for n in range(modes):
            for c in range(3):
                by[n,c] += z[n,c]
                b3[n,c] += h*z[n,c]
        gradient[index] += control_inner(b3, y3)
        reduced_rhs(b3, ux, c2, decay, coupling, z, True)
        for n in range(modes):
            for c in range(3):
                by[n,c] += z[n,c]
                b2[n,c] += h/2*z[n,c]
        gradient[index] += control_inner(b2, y2)
        reduced_rhs(b2, ux, c2, decay, coupling, z, True)
        for n in range(modes):
            for c in range(3):
                by[n,c] += z[n,c]
                b1[n,c] += h/2*z[n,c]
        gradient[index] += control_inner(b1, y)
        reduced_rhs(b1, ux, c1, decay, coupling, z, True)
        for n in range(modes):
            for c in range(3):
                lam[n,c] = by[n,c]+z[n,c]
        if j == 0 or indices[j-1] != index:
            boundaries[index] = lam
    return gradient, boundaries


@njit(cache=True)
def rk4_adjoint(u, dt, indices, cosine, decay, coupling, states):
    return boundary_adjoint(u, dt, indices, cosine, decay, coupling, states)[0]


@njit(cache=True)
def time_grid(edges, maximum_dt):
    count = np.ceil(np.diff(edges)/maximum_dt).astype(np.int64)
    total = np.sum(count)
    dt = np.empty(total)
    indices = np.empty(total, np.int64)
    cosine = np.empty((total, 3))
    j = 0
    for k in range(len(count)):
        h = (edges[k+1]-edges[k])/count[k]
        for step in range(count[k]):
            t = edges[k]+step*h
            dt[j] = h
            indices[j] = k
            cosine[j, 0] = np.cos(t)
            cosine[j, 1] = np.cos(t+h/2)
            cosine[j, 2] = np.cos(t+h)
            j += 1
    return dt, indices, cosine


class Problem:
    def __init__(self, edges, modes=32, maximum_dt=.004):
        self.edges = np.ascontiguousarray(edges, dtype=float)
        self.modes = int(modes)
        self.dt, self.indices, self.cosine = time_grid(self.edges, maximum_dt)
        self.decay = np.arange(modes)/TAU
        self.coupling = SIGMA*np.sqrt(np.arange(1, modes))

    def value(self, u):
        final = rk4_forward(np.ascontiguousarray(u), self.dt, self.indices,
                            self.cosine, self.decay, self.coupling, False)[-1]
        return float(final[0, 1]**2+final[0, 2]**2)

    def value_gradient(self, u):
        u = np.ascontiguousarray(u)
        states = rk4_forward(u, self.dt, self.indices, self.cosine, self.decay, self.coupling, True)
        value = float(states[-1, 0, 1]**2+states[-1, 0, 2]**2)
        gradient = rk4_adjoint(u, self.dt, self.indices, self.cosine, self.decay, self.coupling, states)
        return value, gradient















def exact_propagate(u, edges, modes, dense_dt=None):
    """No time stepping approximation for a piecewise constant control.

    Auxiliary a=r*cos(t), b=r*sin(t) turn the time dependent cos(t) source
    into an autonomous sparse system on (s,a,b), exactly as the existing check.
    """
    G0, Gx, _, trace = _hierarchy_generators(modes, E, SIGMA, TAU)
    size = 3*modes
    state = np.zeros(3*size)
    state[size] = 1.
    records = [(0., 0., state.copy())] if dense_dt else []
    for ux, a, b in zip(u, edges[:-1], edges[1:]):
        G = G0 + ux*Gx if ux else G0
        h = float(b-a)
        if dense_dt:
            count = max(1, math.ceil(h/dense_dt))
            points = expm_multiply(G, state, start=0., stop=h, num=count+1, traceA=trace)
            records.extend((a+h*k/count, float(ux), points[k]) for k in range(1, count+1))
            state = points[-1].copy()
        else:
            state = expm_multiply(h*G, state, traceA=h*trace)
    T = float(edges[-1])
    r = state[size:2*size]*np.cos(T) + state[2*size:]*np.sin(T)
    s = state[:size]
    rho = np.eye(2)/2 + operator_of(r[:3])
    derivative = operator_of(s[:3])
    qfi = qfi_sld(rho, derivative)[0]
    return qfi, r.reshape(modes, 3), s.reshape(modes, 3), records


def basic_checks():
    problem = Problem(np.linspace(0., .73, 8), modes=24, maximum_dt=.004)
    u = np.random.default_rng(901).uniform(-.8, .8, 7)
    started = time.perf_counter()
    value, gradient = problem.value_gradient(u)
    elapsed = time.perf_counter()-started
    numerical = np.array([(problem.value(u+np.eye(7)[j]*1e-5)-problem.value(u-np.eye(7)[j]*1e-5))/2e-5 for j in range(7)])
    error = float(np.max(abs(numerical-gradient)))
    exact, r, s, _ = exact_propagate(u, problem.edges, 24)
    assert error < 2e-7, (gradient, numerical)
    assert abs(value-exact) < 1e-7, (value, exact)
    assert np.max(abs(np.r_[r[0, 1:], s[0, 0]])) < 1e-10
    print(f"Adjoint gradient check: max error {error:.3g}; RK4/exponential QFI error {abs(value-exact):.3g}; compile/check {elapsed:.2f}s", flush=True)
    return {"gradient_max_absolute_error": error, "rk4_exponential_qfi_error": abs(value-exact)}


def analytic_coherence(times):
    z = np.asarray(times)/TAU
    return np.exp(-SIGMA**2*TAU**2*(z+np.expm1(-z)))


def analytic_cpmg(T, pulses, alternating=False):
    """Finite-width pi pulses, generated analytically, not from saved data."""
    if pulses == 0:
        return np.zeros(1), np.array([0., T])
    width = np.pi/E
    centers = (np.arange(pulses)+.5)*T/pulses
    raw = np.sort(np.clip(np.r_[0., centers-width/2, centers+width/2, T], 0., T))
    edges = raw[np.r_[True, np.diff(raw) > 1e-12]]
    midpoint = (edges[:-1]+edges[1:])/2
    on = abs(midpoint[:, None]-centers) <= width/2+1e-12
    signs = (-1.)**np.arange(pulses) if alternating else np.ones(pulses)
    return on @ signs, edges


def average_on_grid(u, old_edges, edges):
    overlap = np.maximum(0., np.minimum(edges[1:, None], old_edges[None, 1:])
                         - np.maximum(edges[:-1, None], old_edges[None, :-1]))
    return np.clip(overlap @ u/np.diff(edges), -1., 1.)


def publish(temporary,target):
    for attempt in range(40):
        try:
            temporary.replace(target)
            return
        except PermissionError:
            if attempt==39:raise
            time.sleep(min(.025*(attempt+1),.25))


def json_write(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    publish(temporary,path)


def atomic_npz(path,**arrays):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    publish(temporary,path)


def compress_control(u,edges):
    # Merge only exactly identical adjacent intervals; no waveform approximation.
    starts=np.r_[0,np.flatnonzero(np.diff(u)!=0.)+1]
    return np.asarray(u)[starts],np.r_[np.asarray(edges)[starts],edges[-1]]


def exponential(u,edges,modes,dense_dt=None):
    v,b=compress_control(u,edges)
    return exact_propagate(v,b,modes,dense_dt)


def new_guesses(phase,edges,seed):
    T=phase*np.pi;rng=np.random.default_rng(seed)
    cp,cp_edges=analytic_cpmg(T,phase)
    reference=average_on_grid(cp,cp_edges,edges)
    alt,alt_edges=analytic_cpmg(T,phase,True)
    guesses=[reference,average_on_grid(alt,alt_edges,edges),np.ones(len(reference))]
    labels=['new_analytic_signal_matched_CPMG','new_analytic_alternating_CPMG','new_analytic_constant_x']
    families=['signal','alternating','broad']
    midpoint=(edges[1:]+edges[:-1])/2;x=midpoint/T
    for length in (1,2,3,4,6,8):
        window=length*np.pi
        if window+.25>=T:continue
        # Protected prefix followed by an analytically generated sensing window.
        u=np.ones(len(reference));late=midpoint>=T-window
        u[late]=reference[late]
        guesses.append(u);labels.append(f'new_analytic_protected_tail_{length}pi');families.append('tail')
        # Early sensing followed by a pi/2 x rotation to store population sensitivity.
        stop=window+np.pi/(2*E)
        raw=np.sort(np.r_[cp_edges[cp_edges<window],window,stop,T])
        raw=raw[np.r_[True,np.diff(raw)>1e-12]]
        mid=(raw[1:]+raw[:-1])/2
        base=cp[np.clip(np.searchsorted(cp_edges,mid,side='right')-1,0,len(cp)-1)].copy()
        base[(mid>=window)&(mid<stop)]=1.;base[mid>=stop]=0.
        guesses.append(average_on_grid(base,raw,edges))
        labels.append(f'new_analytic_early_storage_{length}pi');families.append('storage')
    for k in range(12):
        field=sum(rng.normal()*np.sin(j*np.pi*x+rng.uniform(0.,2*np.pi))/j for j in range(1,11))
        field/=max(np.max(abs(field)),1e-15)
        u=np.clip(reference+(.2+.12*k)*field,-1.,1.) if k<6 else np.clip((.7+.1*(k%4))*field,-1.,1.)
        guesses.append(u);labels.append(f'new_random_fourier_{k+1:02d}');families.append('random')
    centers=np.arange(phase)*np.pi+np.pi/2
    for k in range(8):
        c=centers+rng.normal(0.,(.02+.014*k)*np.pi,phase)
        width=np.pi/E*rng.uniform(.7,1.3,phase)
        sign=np.ones(phase) if k<4 else rng.choice(np.array([-1.,1.]),phase)
        overlap=np.maximum(0.,np.minimum(edges[1:,None],c+width/2)-np.maximum(edges[:-1,None],c-width/2))
        guesses.append(np.clip(overlap@sign/np.diff(edges),-1.,1.))
        labels.append(f'new_random_jittered_pulses_{k+1:02d}');families.append('random_pulses')
    unique=[];unique_labels=[];unique_families=[]
    for u,label,family in zip(guesses,labels,families):
        if not any(np.allclose(u,v,atol=1e-12,rtol=0.) for v in unique):
            unique.append(u);unique_labels.append(label);unique_families.append(family)
    return np.asarray(unique),unique_labels,unique_families,cp,cp_edges


def choose_starts(scores,families,count):
    order=list(map(int,np.argsort(-scores)));selected=order[:min(3,count)]
    for family in ('signal','alternating','tail','storage','broad','random','random_pulses'):
        if len(selected)>=count:break
        candidates=[i for i in order if families[i]==family and i not in selected]
        if candidates:selected.append(candidates[0])
    selected.extend(i for i in order if i not in selected)
    return selected[:count]


def mode_check(u,edges):
    previous=None;history=[];successes=0
    for modes in MODE_COUNTS:
        F,r,s,_=exponential(u,edges,modes)
        low=np.r_[r[:2].ravel(),s[:2].ravel()]
        entry={'modes':modes,'QFI':F,'Bloch_norm':float(np.linalg.norm(r[0]))}
        if previous is not None:
            entry['QFI_change']=abs(F-previous[0])
            entry['low_modes_change']=float(max(abs(low-previous[1])))
            good=max(entry['QFI_change'],entry['low_modes_change'])<max(1e-9,1e-10*abs(F))
            successes=successes+1 if good else 0
        history.append(entry)
        if modes>=32 and successes>=2 and entry['Bloch_norm']<=1.+1e-9:
            return F,modes,history,r,s
        previous=F,low
    raise RuntimeError(f'Layer convergence failed for T/pi={edges[-1]/np.pi:g}: {history}')


def free_curve(output,max_pi):
    _,_,_,records=exponential(np.zeros(1),np.array([0.,max_pi*np.pi]),64,dense_dt=.04)
    rows=[]
    for t,_,state in records:
        size=3*64
        r=(state[size:2*size]*np.cos(t)+state[2*size:]*np.sin(t)).reshape(64,3)[0]
        s=state[:size].reshape(64,3)[0]
        F=qfi_sld(np.eye(2)/2+operator_of(r),operator_of(s))[0]
        rows.append([t,F,r[0]])
    curve=np.asarray(rows);analytic=analytic_coherence(curve[:,0])
    error=float(max(abs(curve[:,2]-analytic)))
    qfi_error=float(max(abs(curve[:,1]-np.sin(curve[:,0])**2*analytic**2)))
    if max(error,qfi_error)>1e-9:raise RuntimeError('Free Hermite solution disagrees with exact OU identity.')
    np.savetxt(output/'no_control_Hermite_curve.csv',curve,delimiter=',',comments='',header='omega_t,QFI,rx')
    return curve,{'modes':64,'coherence_error_vs_analytic':error,'QFI_error_vs_analytic':qfi_error}


def plot_results(output,rows,curve,max_pi):
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':10,
        'axes.labelsize':12,'axes.linewidth':.75,'xtick.direction':'in','ytick.direction':'in',
        'xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(6.4,3.6));fig.subplots_adjust(left=.10,right=.98,bottom=.17,top=.87)
    times=np.r_[0.,[r['omega_T'] for r in rows]]
    ax.plot(curve[:,0],curve[:,1],color=COLORS['no_control'],ls='-.',lw=1.05,label='No control')
    for key,name,marker,ls in (('variational_QFI','optimized','o','-'),('CPMG_QFI','cpmg','s','--')):
        ax.plot(times,np.r_[0.,[r[key] for r in rows]],color=COLORS[name],ls=ls,lw=1.2,
            marker=marker,ms=3.6,mew=.8,mfc='white' if name=='cpmg' else COLORS[name],
            label='Variational' if name=='optimized' else 'CPMG')
    ax.text(.025,.95,r'$\sigma_\xi/\omega=0.8,\quad\omega\tau_c=\pi,\quad E/\omega=8$',
        transform=ax.transAxes,va='top',fontsize=10)
    step=4 if max_pi>=16 else 2 if max_pi>=6 else 1
    ticks=np.unique(np.r_[np.arange(0,max_pi+1,step),max_pi])
    ax.set_xticks(ticks*np.pi,[r'$0$' if k==0 else r'$\pi$' if k==1 else rf'${k}\pi$' for k in ticks])
    ax.set_xticks(np.arange(max_pi+1)*np.pi,minor=True)
    ax.set_xlim(-.22*np.pi,(max_pi+.22)*np.pi)
    ymax=max(max(r['variational_QFI'],r['CPMG_QFI']) for r in rows)*1.18
    ax.set_ylim(-.02*ymax,ymax)
    ax.set_xlabel(r'$\omega T$');ax.set_ylabel(r'$F_\theta$')
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,handlelength=2.4,columnspacing=2.)
    for suffix in ('png','pdf','svg','eps'):
        fig.savefig(output/f'qfi_hermite_x_only_{max_pi}pi.{suffix}',dpi=600,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)





_AUDIT=False
_READS=set()
_DENIED=set()
def input_guard(output,seed):
    global _AUDIT
    if _AUDIT:return
    allowed=[Path(output).resolve()]
    if seed:allowed.append(Path(seed).resolve())
    source=Path(__file__).resolve()
    def audit(event,args):
        if event!='open' or not args or isinstance(args[0],int):return
        try:path=Path(os.fsdecode(args[0])).resolve()
        except (OSError,ValueError,TypeError):return
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        reading=('r' in mode or '+' in mode) if isinstance(mode,str) else not(flags&os.O_WRONLY)
        if reading and path.is_relative_to(ROOT):
            _READS.add(str(path))
            if path!=source and not any(path.is_relative_to(a) for a in allowed):
                _DENIED.add(str(path));raise PermissionError(f'Unrelated workspace input forbidden: {path}')
    sys.addaudithook(audit);_AUDIT=True

def save_audit(out,label):
    json_write(Path(out)/f'input_audit_{label}.json',{'workspace_reads':sorted(_READS),
        'denied_read_attempts':sorted(_DENIED),'Monte_Carlo_used':False,'noise_samples':0})

@njit(cache=True)
def edge_gradients(u,lam,states,starts):
    result=np.empty(len(u)-1)
    for j in range(1,len(u)):
        result[j-1]=(u[j-1]-u[j])*control_inner(lam[j],states[starts[j]])
    return result

"""Exact discrete RK4 derivatives of both amplitudes and interval endpoints."""

@njit(cache=True, inline='always')
def source_inner(adjoint,y):
    value=0.
    for n in range(len(y)):
        if n%2:value-=adjoint[n,0]*y[n,1]
        else:value+=adjoint[n,1]*y[n,0]
    return value


@njit(cache=True)
def fixed_time_grid(edges,counts):
    total=np.sum(counts)
    dt=np.empty(total);indices=np.empty(total,np.int64);cs=np.empty((total,6))
    j=0
    for k in range(len(counts)):
        h=(edges[k+1]-edges[k])/counts[k]
        for step in range(counts[k]):
            t=edges[k]+step*h
            dt[j]=h;indices[j]=k
            cs[j,0]=np.cos(t);cs[j,1]=np.cos(t+h/2);cs[j,2]=np.cos(t+h)
            cs[j,3]=np.sin(t);cs[j,4]=np.sin(t+h/2);cs[j,5]=np.sin(t+h)
            j+=1
    return dt,indices,cs


@njit(cache=True)
def exact_time_adjoint(u,dt,indices,cs,decay,coupling,states,counts):
    modes=len(decay);P=len(u)
    gradient=np.zeros(P);edges_gradient=np.zeros(P+1)
    lam=np.zeros((modes,3));lam[0,1:3]=2*states[-1,0,1:3]
    begins=np.empty(P+1,np.int64);begins[0]=0
    for k in range(P):begins[k+1]=begins[k]+counts[k]
    k1,k2,k3,k4,y2,y3,y4,b1,b2,b3,b4,by,z=[np.empty_like(lam) for _ in range(13)]
    for j in range(len(dt)-1,-1,-1):
        h,index=dt[j],indices[j];ux,y=u[index],states[j]
        c1,c2,c4=cs[j,0],cs[j,1],cs[j,2]
        reduced_rhs(y,ux,c1,decay,coupling,k1)
        for n in range(modes):
            for c in range(3):y2[n,c]=y[n,c]+h/2*k1[n,c]
        reduced_rhs(y2,ux,c2,decay,coupling,k2)
        for n in range(modes):
            for c in range(3):y3[n,c]=y[n,c]+h/2*k2[n,c]
        reduced_rhs(y3,ux,c2,decay,coupling,k3)
        for n in range(modes):
            for c in range(3):
                y4[n,c]=y[n,c]+h*k3[n,c]
                b1[n,c]=h/6*lam[n,c];b2[n,c]=h/3*lam[n,c]
                b3[n,c]=h/3*lam[n,c];b4[n,c]=h/6*lam[n,c];by[n,c]=lam[n,c]
        reduced_rhs(y4,ux,c4,decay,coupling,k4)
        bh=0.
        for n in range(modes):
            for c in range(3):bh+=lam[n,c]*(k1[n,c]+2*k2[n,c]+2*k3[n,c]+k4[n,c])/6
        phase=-cs[j,5]*source_inner(b4,y4);bt=phase;bh+=phase
        gradient[index]+=control_inner(b4,y4)
        reduced_rhs(b4,ux,c4,decay,coupling,z,True)
        for n in range(modes):
            for c in range(3):
                bh+=z[n,c]*k3[n,c];by[n,c]+=z[n,c];b3[n,c]+=h*z[n,c]
        phase=-cs[j,4]*source_inner(b3,y3);bt+=phase;bh+=phase/2
        gradient[index]+=control_inner(b3,y3)
        reduced_rhs(b3,ux,c2,decay,coupling,z,True)
        for n in range(modes):
            for c in range(3):
                bh+=z[n,c]*k2[n,c]/2;by[n,c]+=z[n,c];b2[n,c]+=h/2*z[n,c]
        phase=-cs[j,4]*source_inner(b2,y2);bt+=phase;bh+=phase/2
        gradient[index]+=control_inner(b2,y2)
        reduced_rhs(b2,ux,c2,decay,coupling,z,True)
        for n in range(modes):
            for c in range(3):
                bh+=z[n,c]*k1[n,c]/2;by[n,c]+=z[n,c];b1[n,c]+=h/2*z[n,c]
        bt-=cs[j,3]*source_inner(b1,y)
        gradient[index]+=control_inner(b1,y)
        reduced_rhs(b1,ux,c1,decay,coupling,z,True)
        for n in range(modes):
            for c in range(3):lam[n,c]=by[n,c]+z[n,c]
        fraction=(j-begins[index])/counts[index]
        edges_gradient[index]+=bt*(1-fraction)-bh/counts[index]
        edges_gradient[index+1]+=bt*fraction+bh/counts[index]
    return gradient,edges_gradient[1:-1]


class AdaptiveProblem:
    def __init__(self,segments,T,floor,reference,modes=32,maximum_dt=.002):
        self.P=int(segments);self.T=float(T);self.floor=float(floor);self.reference=int(reference)
        self.modes=int(modes);self.maximum_dt=float(maximum_dt);self.counts=None;self.objective_scale=None
        self.mask=np.arange(self.P)!=self.reference
        self.lower=np.r_[-np.ones(self.P),-25*np.ones(self.P-1)]
        self.upper=-self.lower

    def pack(self,u,edges):
        weights=np.maximum(np.diff(edges)-self.floor,1e-15)
        logs=np.log(weights/weights[self.reference])
        return np.r_[np.clip(u,-1.,1.),np.clip(logs[self.mask],-25.,25.)]

    def decode(self,x):
        logs=np.zeros(self.P);logs[self.mask]=x[self.P:]
        weights=np.exp(logs-logs.max());weights/=weights.sum()
        durations=self.floor+(self.T-self.P*self.floor)*weights
        edges=np.r_[0.,np.cumsum(durations)];edges[-1]=self.T
        return np.ascontiguousarray(x[:self.P]),edges,weights

    def value_gradient(self,x):
        u,edges,weights=self.decode(x)
        widths=np.diff(edges)
        if np.min(widths)<=0:raise RuntimeError('Nonpositive adaptive duration')
        if self.counts is None:self.counts=np.maximum(1,np.ceil(widths/self.maximum_dt).astype(np.int64))
        counts=self.counts.copy()
        grow=widths/counts>2*self.maximum_dt
        counts[grow]=np.ceil(widths[grow]/self.maximum_dt).astype(np.int64)
        dt,index,cs=fixed_time_grid(edges,counts)
        decay=np.arange(self.modes)/TAU;coupling=SIGMA*np.sqrt(np.arange(1,self.modes))
        states=rk4_forward(u,dt,index,cs,decay,coupling,True)
        F=float(np.sum(states[-1,0,1:3]**2))
        if self.objective_scale is None:self.objective_scale=max(1.,abs(F))
        gu,gb=exact_time_adjoint(u,dt,index,cs,decay,coupling,states,counts)
        gd=np.r_[np.cumsum(gb[::-1])[::-1],0.]
        gz=(self.T-self.P*self.floor)*weights*(gd-np.dot(weights,gd))
        return F,np.r_[gu,gz[self.mask]]

    def residual(self,x,g):
        return float(max(abs(np.clip(x+g/max(1.,self.objective_scale or 1.),self.lower,self.upper)-x)))


def adaptive_check():
    edges=np.linspace(0.,.83,8);u=np.random.default_rng(912).uniform(-.7,.7,7)
    p=AdaptiveProblem(7,edges[-1],1e-8,2,modes=24,maximum_dt=.002)
    x=p.pack(u,edges);F,g=p.value_gradient(x);errors=[]
    for j in (0,3,7,9,12):
        shift=np.zeros(len(x));shift[j]=1e-5
        numerical=(p.value_gradient(x+shift)[0]-p.value_gradient(x-shift)[0])/2e-5
        errors.append(abs(numerical-g[j]))
    assert max(errors)<2e-7,errors
    decoded,b,_=p.decode(x);exact=exponential(decoded,b,24)[0]
    assert abs(exact-F)<1e-8
    report={'maximum_adaptive_gradient_FD_error':float(max(errors)),'RK4_exponential_QFI_error':abs(exact-F)}
    print('Adaptive checks:',json.dumps(report),flush=True);return report

def saturated_compaction(u,edges,gradient):
    saturated=((u==1.)&(gradient>=-1e-10))|((u==-1.)&(gradient<=1e-10))
    merge=(u[1:]==u[:-1])&saturated[1:]&saturated[:-1]
    starts=np.r_[0,np.flatnonzero(~merge)+1]
    if len(u)-len(starts)<max(4,.15*len(u)):return None
    return u[starts].copy(),np.r_[edges[starts],edges[-1]]

def optimize_adaptive(u,edges,path,label,options,level):
    path=Path(path);started=last=time.perf_counter();T=float(edges[-1]);P=len(u)
    p=AdaptiveProblem(P,T,1e-8/2**level,int(np.argmax(np.diff(edges))))
    x=p.pack(u,edges);initial=x.copy();resumed=False;compactions=[]
    if options['resume'] and path.exists():
        with np.load(path) as d:
            p=AdaptiveProblem(int(d['segments']),float(d['T']),float(d['floor']),int(d['reference']))
            x=d['params'].copy();initial=d['initial'].copy()
            if 'compaction_log' in d:compactions=json.loads(str(d['compaction_log']))
        resumed=True
        if path.with_suffix('.json').exists():
            old=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if old['stationarity_passed']:
                u,b,_=p.decode(x);return u,b,old
    F,g=p.value_gradient(x);best,bestF,bestg=x.copy(),F,g.copy()
    evaluations=1;history=[];statuses=[];tol=options['gradient_tolerance'];last_compaction_check=0;force_lbfgs=False
    def checkpoint():
        u,b,_=p.decode(best)
        atomic_npz(path,params=best,gradient=bestg,initial=initial,control=u,edges=b,
            segments=p.P,T=p.T,floor=p.floor,reference=p.reference,compaction_log=json.dumps(compactions))
    class Stationary(Exception):pass
    class CompactRepresentation(Exception):pass
    class FastBudgetReached(Exception):pass
    for restart in range(options['restarts']):
        if p.residual(best,bestg)<=tol:break
        # This is only initialization cleanup. Refined meshes keep all newly
        # introduced amplitude degrees of freedom for the convergence test.
        if level==0:
            old_u,old_edges,_=p.decode(best)
            candidate=saturated_compaction(old_u,old_edges,bestg[:p.P])
            if candidate is not None:
                checkpoint()
                archive=path.with_name(path.stem+f'_before_compact_{time.time_ns()}.npz')
                shutil.copy2(path,archive)
                old_P,old_F=p.P,bestF;new_u,new_edges=candidate
                new_floor=min(p.floor/10.,float(min(np.diff(new_edges)))/100.)
                new_p=AdaptiveProblem(len(new_u),p.T,new_floor,int(np.argmax(np.diff(new_edges))))
                new_x=new_p.pack(new_u,new_edges);decoded,new_b,_=new_p.decode(new_x)
                edge_error=float(max(abs(new_b-new_edges)))
                if edge_error>1e-9*max(1.,p.T):raise RuntimeError('Compaction re-encoding changed pulse boundaries.')
                new_F,new_g=new_p.value_gradient(new_x)
                if abs(new_F-old_F)>1e-6*max(1.,abs(old_F)):raise RuntimeError('Compaction changed the physical QFI.')
                compactions.append({'old_segments':old_P,'new_segments':len(new_u),'archive':archive.name,
                    'identical_adjacent_saturated_controls_only':True,'maximum_boundary_roundoff':edge_error,
                    'QFI_before':old_F,'QFI_after':new_F})
                p=new_p;best=new_x.copy();initial=new_x.copy();bestF,bestg=new_F,new_g
                force_lbfgs=False;checkpoint()
                print(f'{label}: compacted identical saturated segments {old_P} -> {p.P}; QFI change={new_F-old_F:.3g}',flush=True)
        def objective(x):
            nonlocal best,bestF,bestg,evaluations,last
            F,g=p.value_gradient(x);evaluations+=1;pg=p.residual(x,g)
            if F>bestF or (F>=bestF-2e-13*max(1.,abs(bestF)) and pg<p.residual(best,bestg)):
                best,bestF,bestg=x.copy(),float(F),g.copy()
            now=time.perf_counter();history.append([evaluations,F,bestF,pg,now-started,restart])
            if evaluations==2 or now-last>25.:
                checkpoint();print(f'{label}: eval {evaluations}, QFI={bestF:.10f}, parameter gradient={pg:.3g}, {now-started:.1f}s',flush=True);last=now
            if options.get('fast') and evaluations>=options['evaluation_budget']:raise FastBudgetReached()
            return -F,-g
        def callback(x):
            nonlocal last_compaction_check
            if p.residual(best,bestg)<=tol:raise Stationary()
            if level==0 and evaluations-last_compaction_check>=100:
                last_compaction_check=evaluations
                cu,cb,_=p.decode(best)
                if saturated_compaction(cu,cb,bestg[:p.P]) is not None:raise CompactRepresentation()
        use_slsqp=(restart%2==1 or (restart==0 and (resumed or compactions or options.get('fast')))) and len(best)<=420 and not force_lbfgs
        force_lbfgs=False
        method='SLSQP' if use_slsqp else 'L-BFGS-B'
        try:
            result=minimize(objective,best,jac=True,method=method,bounds=list(zip(p.lower,p.upper)),callback=callback,
                options={'maxiter':options['iterations'],'ftol':2e-14,'disp':False} if use_slsqp else
                    {'maxiter':options['iterations'],'maxfun':4*options['iterations'],'ftol':1e-15,'gtol':1e-9,'maxcor':40,'maxls':30})
            statuses.append({'method':method,'message':str(result.message),'iterations':int(result.nit)})
            if use_slsqp and p.residual(best,bestg)>tol:force_lbfgs=True
        except Stationary:statuses.append({'method':method,'message':'projected parameter gradient reached'})
        except CompactRepresentation:statuses.append({'method':method,'message':'restart after merging exactly identical saturated segments'})
        except FastBudgetReached:
            statuses.append({'method':method,'message':'user-requested fast evaluation budget reached'})
            break
    F,g=p.value_gradient(best);bestF,bestg=F,g;checkpoint();u,b,_=p.decode(best);pg=p.residual(best,g)
    report={'label':label,'QFI_RK4':F,'projected_gradient':pg,'stationarity_passed':pg<=tol,
        'gradient_tolerance':tol,'gradient_scale_QFI':float(p.objective_scale or 1.),'relative_projected_gradient':pg,'raw_projected_gradient':float(max(abs(np.clip(best+g,p.lower,p.upper)-best))),'parameterization':'bounded amplitudes and bounded log-duration ratios',
        'parameters':len(best),'segments':len(u),'reference_duration_index':p.reference,'duration_floor':p.floor,
        'resumed_checkpoint':resumed,'evaluations':evaluations,'solver_runs':statuses,'initial_representation_compactions':compactions,
        'finite_budget_fast_mode':bool(options.get('fast')),
        'wall_seconds':time.perf_counter()-started}
    json_write(path.with_suffix('.json'),report)
    np.savetxt(path.with_suffix('.csv'),np.asarray(history).reshape(-1,6),delimiter=',',comments='',
        header='evaluation,QFI,QFI_best,projected_parameter_gradient,seconds,restart')
    print(f'{label}: DONE F={F:.10f}, parameter gradient={pg:.3g}, pass={pg<=tol}',flush=True)
    return u,b,report

def initial_seed(phase,point,seed_dir,options):
    candidates=[];provenance=[];T=phase*np.pi
    if seed_dir:
        folder=Path(seed_dir)/f'T_{phase:02d}pi'
        files=[]
        for pattern in ('controls.npz','adaptive_level_?.npz','active_sqp_diagnostic.npz','coarse_*.npz','fine_*.npz','grid_*.npz'):files.extend(folder.glob(pattern))
        diagnostic=Path(seed_dir)/f'diagnostic_adaptive_T{phase:02d}pi.npz'
        if diagnostic.exists():files.append(diagnostic)
        for file in files:
            with np.load(file,allow_pickle=False) as d:
                if 'optimized' in d:
                    u,b=d['optimized'][:,0].copy(),d['optimized_edges'].copy()
                elif 'control' in d and 'edges' in d:
                    u,b=d['control'].copy(),d['edges'].copy()
                else:continue
            if abs(b[-1]-T)>1e-10 or max(abs(u))>1.+1e-9:raise RuntimeError('Seed control parameter mismatch')
            u=np.clip(u,-1.,1.);u,b=compress_control(u,b)
            F=Problem(b,modes=32,maximum_dt=.006).value(u)
            origin={'path':str(file.resolve()),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'screening_QFI':F}
            candidates.append((F,u,b,origin));provenance.append(origin)
    if not candidates:
        P=max(96,48*phase);edges=np.linspace(0.,T,P+1)
        seed=(options['control_seed']+phase*10939)%(1<<62)
        guesses,labels,families,_,_=new_guesses(phase,edges,seed)
        problem=Problem(edges,modes=24,maximum_dt=.012)
        scores=np.array([problem.value(u) for u in guesses]);selected=choose_starts(scores,families,options['starts'])
        atomic_npz(point/'fresh_initials.npz',controls=guesses,edges=edges,labels=np.asarray(labels),scores=scores,selected_indices=selected,seed=seed)
        for k,index in enumerate(selected):
            def objective(u):
                F,g=problem.value_gradient(u);return -F,-g
            result=minimize(objective,guesses[index],jac=True,method='L-BFGS-B',bounds=[(-1.,1.)]*P,
                options={'maxiter':500,'ftol':1e-13,'gtol':1e-6,'maxcor':40})
            u=np.clip(result.x,-1.,1.);u,b=compress_control(u,edges)
            F=Problem(b,modes=32,maximum_dt=.006).value(u)
            origin={'fresh_control_seed':seed,'label':labels[index],'selected_index':int(index),'screening_QFI':F}
            candidates.append((F,u,b,origin));provenance.append(origin)
            atomic_npz(point/f'fresh_coarse_{k+1:02d}.npz',control=u,edges=b,initial=guesses[index])
    candidates.sort(key=lambda a:a[0],reverse=True);F,u,b,origin=candidates[0]
    atomic_npz(point/'adaptive_initial_control.npz',control=u,edges=b)
    json_write(point/'seed_provenance.json',{'all_inputs_from_current_20pi_Hermite_task':bool(seed_dir),
        'Monte_Carlo_inputs':False,'selected':origin,'candidates':provenance})
    return u,b

def solve_point(job):
    phase,out_name,seed_name,options=job;out=Path(out_name);input_guard(out,seed_name)
    point=out/f'T_{phase:02d}pi';point.mkdir(exist_ok=True)
    if options['resume'] and (point/'result.json').exists():
        old=json.loads((point/'result.json').read_text(encoding='utf-8'))
        converged=old.get('optimization_convergence_met',old['stationarity_passed'] and old['control_grid_passed'])
        if old['passed'] and (converged or options.get('fast')):return old
    started=time.perf_counter();T=phase*np.pi
    if options['resume'] and (point/'adaptive_initial_control.npz').exists():
        with np.load(point/'adaptive_initial_control.npz') as d:u,b=d['control'].copy(),d['edges'].copy()
    else:u,b=initial_seed(phase,point,seed_name,options)
    print(f'T={phase}pi adaptive initial segments={len(u)}',flush=True)
    history=[];grid_pass=False;previous=None;previous_pass=False
    for level in range(options['refinements']+1):
        if level:
            boundaries=np.column_stack((b[:-1],(b[:-1]+b[1:])/2)).ravel()
            b=np.r_[boundaries,b[-1]];u=np.repeat(u,2)
        u,b,report=optimize_adaptive(u,b,point/f'adaptive_level_{level}.npz',f'T={phase}pi adaptive level {level}',options,level)
        tolerance=max(5e-4,5e-4*abs(report['QFI_RK4']))
        report.update(mesh_tolerance=tolerance,mesh_gain=None if previous is None else report['QFI_RK4']-previous)
        json_write(point/f'adaptive_level_{level}.json',report);history.append(report)
        if previous is not None and previous_pass and report['stationarity_passed'] and abs(report['mesh_gain'])<tolerance:
            grid_pass=True;break
        previous=report['QFI_RK4'];previous_pass=report['stationarity_passed']
    F,modes,layers,r,s=mode_check(u,b)
    cp,cp_edges=analytic_cpmg(T,phase);Fcp,cp_modes,cp_layers,_,_=mode_check(cp,cp_edges)
    time_checks=[]
    for dt in (.004,.002,.001):
        value=Problem(b,modes=max(48,modes),maximum_dt=dt).value(u)
        time_checks.append({'maximum_dt':dt,'QFI':value,'error_vs_exponential':abs(value-F)})
    time_pass=time_checks[-1]['error_vs_exponential']<max(1e-7,1e-8*abs(F))
    atomic_npz(point/'controls.npz',optimized=np.column_stack((u,np.zeros(len(u)))),optimized_edges=b,
        cpmg=np.column_stack((cp,np.zeros(len(cp)))),cpmg_edges=cp_edges)
    atomic_npz(point/'hierarchy_final.npz',r=r,s=s,modes=modes)
    for name,v,edges in (('optimized',u,b),('cpmg',cp,cp_edges)):
        np.savetxt(point/f'control_{name}.csv',np.column_stack((edges[:-1],edges[1:],E*v,np.zeros(len(v)))),
            delimiter=',',comments='',header='omega_t_start,omega_t_stop,ux_over_omega,uy_over_omega')
    row={'T_over_pi':phase,'omega_T':T,'variational_QFI':F,'CPMG_QFI':Fcp,'no_control_QFI':0.,
        'Hermite_modes':modes,'CPMG_modes':cp_modes,'Hermite_convergence':layers,'CPMG_convergence':cp_layers,
        'time_step_checks':time_checks,'time_step_passed':bool(time_pass),'control_grid_history':history,
        'control_grid_passed':bool(grid_pass),'projected_gradient':report['projected_gradient'],'gradient_scale_QFI':report.get('gradient_scale_QFI',1.),'raw_projected_gradient':report.get('raw_projected_gradient',report['projected_gradient']),
        'stationarity_passed':report['stationarity_passed'],'control_intervals':len(u),
        'optimizer_parameterization':'adaptive pulse amplitudes and log-duration ratios',
        'stationarity_metric':'box-projected gradient of QFI divided by its starting magnitude max(1,F_start), in normalized amplitudes and log-duration-ratio parameters',
        'uniform_grid_functional_stationarity_claimed':False,'noise_samples':0,'Monte_Carlo_used':False,
        'r0':r[0].tolist(),'dq_r0':s[0].tolist(),'global_optimum_proven':False,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'wall_seconds':time.perf_counter()-started}
    row['optimization_convergence_met']=bool(grid_pass and report['stationarity_passed'])
    row['finite_budget_fast_mode']=bool(options.get('fast'))
    row['passed']=bool(time_pass and (row['optimization_convergence_met'] or options.get('fast')))
    json_write(point/'result.json',row);save_audit(out,f'T{phase:02d}pi')
    print(f'T={phase}pi FINAL F={F:.10f}, CPMG={Fcp:.10f}, pass={row["passed"]}, {row["wall_seconds"]:.1f}s',flush=True)
    return row

def main():
    parser=argparse.ArgumentParser(description=__doc__,parents=[_early])
    parser.add_argument('--fresh',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--fast',action='store_true');parser.add_argument('--evaluation-budget',type=int,default=400)
    parser.add_argument('--step-pi',type=int,default=2)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--workers',type=int,default=min(8,os.cpu_count() or 1))
    parser.add_argument('--iterations',type=int,default=2200);parser.add_argument('--restarts',type=int,default=6)
    parser.add_argument('--refinements',type=int,default=3);parser.add_argument('--max-pi',type=int,default=20)
    args=parser.parse_args();sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    print('Python executable:',sys.executable,flush=True);print('Python version:',sys.version,flush=True)
    print('Working directory:',Path.cwd(),flush=True)
    print('Dependencies:',', '.join(f'{p}={sys.modules[p].__version__}' for p in ('numpy','scipy','numba','matplotlib')),flush=True)
    out=RUNTIME_OUTPUT;out.mkdir(parents=True,exist_ok=True)
    seed=None if args.fresh or not args.seed_dir.exists() else args.seed_dir.resolve()
    control_seed=secrets.randbits(61)
    if seed:
        data=json.loads((seed/'run_manifest.json').read_text(encoding='utf-8'))
        assert data['physics']['E_over_omega']==E and data['physics']['sigma_over_omega']==SIGMA and data['physics']['omega_tau_c']==TAU
        assert data['noise_samples']==0
        control_seed=data['options']['control_seed']
    input_guard(out,seed);started=time.perf_counter()
    checks={'Hermite_core':basic_checks(),'adaptive_parameters':adaptive_check()}
    json_write(out/'basic_checks.json',checks)
    if args.self_test:save_audit(out,'self_test');return 0
    options={'resume':args.resume,'iterations':args.iterations,'restarts':args.restarts,'refinements':args.refinements,
        'max_pi':args.max_pi,'control_seed':control_seed,'starts':8,'gradient_tolerance':3e-6,'smoke':args.smoke_test,
        'step_pi':args.step_pi,'fast':args.fast,'evaluation_budget':args.evaluation_budget}
    if args.fast:options.update(refinements=0,restarts=2)
    if args.smoke_test:options.update(max_pi=1,step_pi=1,iterations=8,restarts=1,refinements=1)
    path=out/'run_manifest.json'
    if path.exists():
        if not args.resume:raise RuntimeError('Use --resume or a fresh output directory')
        old=json.loads(path.read_text(encoding='utf-8'))
        assert old['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for key in ('max_pi','control_seed','starts','gradient_tolerance','smoke'):options[key]=old['options'][key]
        assert old['current_task_seed_directory']==(str(seed) if seed else None)
        json_write(out/f'resume_options_{time.time_ns()}.json',options)
    else:
        json_write(path,{'started_at':datetime.now().astimezone().isoformat(),'python':sys.executable,'python_version':sys.version,
            'cwd':str(Path.cwd()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'options':options,
            'physics':{'E_over_omega':E,'sigma_over_omega':SIGMA,'omega_tau_c':TAU,'theta':0.,'initial_state':'+x','uy':0.},
            'method':'parity-reduced OU-Hermite with adaptive bounded pulses; preallocated RK4/discrete adjoint','noise_samples':0,'current_task_seed_directory':str(seed) if seed else None,
            'Monte_Carlo_inputs':False,'global_optimum_proven':False})
        (out/'program_snapshot.py').write_bytes(Path(__file__).read_bytes())
    if options['step_pi']<1 or options['max_pi']%options['step_pi']:raise ValueError('max-pi must be divisible by step-pi')
    jobs=[(n,str(out),str(seed) if seed else None,options) for n in range(options['max_pi'],0,-options['step_pi'])];rows=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,len(jobs)),mp_context=mp.get_context('spawn')) as pool:
        pending={pool.submit(solve_point,j) for j in jobs};last=time.perf_counter()
        while pending:
            done,pending=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for f in done:rows.append(f.result())
            now=time.perf_counter()
            if done or now-last>25.:
                print(f'Adaptive integer-pi cases completed: {len(rows)}/{len(jobs)}',flush=True);last=now
    rows.sort(key=lambda r:r['T_over_pi']);json_write(out/'results.json',rows)
    curve,free_check=free_curve(out,options['max_pi']);plot_results(out,rows,curve,options['max_pi'])
    table=[[0.]*10+[1.,0.]]
    for r in rows:
        table.append([r['T_over_pi'],r['omega_T'],r['variational_QFI'],r['CPMG_QFI'],0.,
            r['Hermite_convergence'][-1].get('QFI_change',0.),r['CPMG_convergence'][-1].get('QFI_change',0.),
            r['projected_gradient'],r['control_intervals'],r['Hermite_modes'],
            float(r.get('optimization_convergence_met',r['stationarity_passed'] and r['control_grid_passed'])),
            float(r.get('finite_budget_fast_mode',False))])
    np.savetxt(out/'QFI_vs_terminal_time_Hermite.csv',table,delimiter=',',comments='',
        header='T_over_pi,omega_T,variational_QFI,CPMG_QFI,no_control_QFI,last_mode_change_variational,last_mode_change_CPMG,projected_adaptive_parameter_gradient,adaptive_segments,Hermite_modes,strict_optimization_convergence_met,finite_budget_fast_mode')
    verification={'passed':all(r['passed'] for r in rows),'cases':len(rows),'free_curve_check':free_check,
        'terminal_points_over_pi':[0]+[r['T_over_pi'] for r in rows],
        'finite_budget_fast_mode':bool(options.get('fast')),
        'all_optimization_convergence_met':all(r.get('optimization_convergence_met',r['stationarity_passed'] and r['control_grid_passed']) for r in rows),
        'all_adaptive_parameter_stationarity_checks_passed':all(r['stationarity_passed'] for r in rows),
        'all_mesh_checks_passed':all(r['control_grid_passed'] for r in rows),
        'all_time_checks_passed':all(r['time_step_passed'] for r in rows),'noise_samples':0,'Monte_Carlo_used':False,
        'uniform_grid_functional_stationarity_claimed':False,'global_optimum_proven':False}
    json_write(out/'verification.json',verification);save_audit(out,'main')
    elapsed=time.perf_counter()-started;json_write(out/'timing_last_run.json',{'wall_seconds':elapsed,'finished_at':datetime.now().astimezone().isoformat()})
    print(f'Completed in {elapsed:.3f}s, checks passed={verification["passed"]}, output={out}',flush=True)
    if not verification['passed'] and not options['smoke']:raise RuntimeError('Unfinished checks; resume this run with larger budgets.')
    return 0

if __name__=='__main__':
    mp.freeze_support();raise SystemExit(main())
