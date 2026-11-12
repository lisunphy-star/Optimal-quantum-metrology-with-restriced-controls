"""Transverse sensing using the manuscript convention S_a = sigma_a/2.

H_theta = [(Delta+ux) sigma_x + uy sigma_y + (theta+uz) sigma_z]/2.
All three controls are allowed, with ux^2+uy^2+uz^2 <= E^2 at every time.
Initial state |+x>, local theta=0, Delta=1, E=.5, terminal times 4 and 40.
The SU(2) exponential and its first/mixed second derivatives give the exact
discrete adjoint for piecewise-constant controls. Independent 4x4 block
exponentials and Frechet derivatives validate the implementation.

Run with python transverse_field_spin_prl.py.
Use --check-only for numerical checks or --replot to render saved results.
"""
from pathlib import Path
from datetime import datetime
import argparse
import hashlib
import json
import math
import os
import sys
import time

ROOT = Path(__file__).resolve().parent
_output_parser = argparse.ArgumentParser(add_help=False)
_output_parser.add_argument('--output-dir', type=Path,
    default=ROOT/'outputs'/'transverse_field_spin_prl_T4_T40')
_output_args, _ = _output_parser.parse_known_args()
OUT = _output_args.output_dir.resolve()
CONDA = Path(sys.executable).resolve().parent
os.environ['PATH'] = os.pathsep.join(str(p) for p in (CONDA, CONDA/'Library'/'bin', CONDA/'Scripts')) + os.pathsep + os.environ.get('PATH', '')
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(OUT/'numba_cache')
os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib_config')
import numpy as np
import scipy
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from numba import njit

SX = np.array([[0,1],[1,0]],complex)
SY = np.array([[0,-1j],[1j,0]],complex)
SZ = np.diag([1.,-1.]).astype(complex)
OPS = np.array([SX,SY,SZ])
SPIN_SCALE = .5
PLUS = np.array([1,1],complex)/np.sqrt(2)


def dump(path, value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


@njit(cache=True, inline='always')
def coefficients(h, dt, scale=.5):
    q = h[0]*h[0]+h[1]*h[1]+h[2]*h[2]
    a = dt*scale
    v2 = a*a*q
    v = math.sqrt(v2)
    c = math.cos(v)
    f = a if v==0. else a*math.sin(v)/v
    if v2<1e-4:
        fp = a**3*(-1/6+v2/60-v2*v2/1680+v2**3/90720)
        fpp = a**5*(1/60-v2/840+v2*v2/30240-v2**3/1995840)
    else:
        fp = (a*c-f)/(2*q)
        fpp = -a*a*f/(4*q)-3*fp/(2*q)
    return c,f,-a*f/2,fp,-a*fp/2,fpp


@njit(cache=True, inline='always')
def hpsi(p,h):
    return h[2]*p[0]+(h[0]-1j*h[1])*p[1],(h[0]+1j*h[1])*p[0]-h[2]*p[1]


@njit(cache=True, inline='always')
def sigma(p,a):
    if a==0:return p[1],p[0]
    if a==1:return -1j*p[1],1j*p[0]
    return p[0],-p[1]


@njit(cache=True, inline='always')
def apply_u(p,h,c,dagger=False):
    hp=hpsi(p,h);sign=1j if dagger else -1j
    return c[0]*p[0]+sign*c[1]*hp[0],c[0]*p[1]+sign*c[1]*hp[1]


@njit(cache=True, inline='always')
def apply_du(p,a,h,c,dagger=False):
    hp=hpsi(p,h);pa=sigma(p,a);sign=1j if dagger else -1j
    return (2*h[a]*c[2]*p[0]+sign*(2*h[a]*c[3]*hp[0]+c[1]*pa[0]),
            2*h[a]*c[2]*p[1]+sign*(2*h[a]*c[3]*hp[1]+c[1]*pa[1]))


@njit(cache=True, inline='always')
def apply_mixed(p,a,h,c):
    hp=hpsi(p,h);pa=sigma(p,a);pz=sigma(p,2)
    delta=1. if a==2 else 0.
    scalar=4*h[a]*h[2]*c[4]+2*delta*c[2]
    prefactor=4*h[a]*h[2]*c[5]+2*delta*c[3]
    return (scalar*p[0]-1j*(prefactor*hp[0]+2*c[3]*(h[a]*pz[0]+h[2]*pa[0])),
            scalar*p[1]-1j*(prefactor*hp[1]+2*c[3]*(h[a]*pz[1]+h[2]*pa[1])))


@njit(cache=True)
def forward(u, Delta, T, theta=0., scale=.5):
    N=len(u);dt=T/N
    states=np.zeros((N+1,4),np.complex128)
    states[0,0]=states[0,1]=1/math.sqrt(2)
    for k in range(N):
        h=np.array([Delta+u[k,0],u[k,1],theta+u[k,2]])
        c=coefficients(h,dt,scale)
        p=(states[k,0],states[k,1]);f=(states[k,2],states[k,3])
        pn=apply_u(p,h,c);fn=apply_u(f,h,c);source=apply_du(p,2,h,c)
        states[k+1,0],states[k+1,1]=pn
        states[k+1,2],states[k+1,3]=fn[0]+source[0],fn[1]+source[1]
    return states


@njit(cache=True)
def value_gradient(u, Delta, T, gradient=True, scale=.5):
    states=forward(u,Delta,T,0.,scale)
    N=len(u);dt=T/N
    p=(states[N,0],states[N,1]);f=(states[N,2],states[N,3])
    overlap=np.conj(p[0])*f[0]+np.conj(p[1])*f[1]
    F=4*(abs(f[0])**2+abs(f[1])**2-abs(overlap)**2)
    g=np.zeros_like(u)
    if not gradient:return F,g
    lp=(-4*np.conj(overlap)*f[0],-4*np.conj(overlap)*f[1])
    lf=(4*(f[0]-overlap*p[0]),4*(f[1]-overlap*p[1]))
    for k in range(N-1,-1,-1):
        h=np.array([Delta+u[k,0],u[k,1],u[k,2]])
        c=coefficients(h,dt,scale)
        p=(states[k,0],states[k,1]);f=(states[k,2],states[k,3])
        for a in range(3):
            dp=apply_du(p,a,h,c);df=apply_du(f,a,h,c);mixed=apply_mixed(p,a,h,c)
            g[k,a]=2*(np.conj(lp[0])*dp[0]+np.conj(lp[1])*dp[1]+
                np.conj(lf[0])*(df[0]+mixed[0])+np.conj(lf[1])*(df[1]+mixed[1])).real
        back_p=apply_u(lp,h,c,True);extra=apply_du(lf,2,h,c,True);back_f=apply_u(lf,h,c,True)
        lp=(back_p[0]+extra[0],back_p[1]+extra[1]);lf=back_f
    return F,g


def project(u,E):
    return u*np.minimum(1.,E/np.maximum(np.linalg.norm(u,axis=-1),1e-300))[...,None]


def validate_saved_case(saved,Delta,E,T):
    """Reject a checkpoint from a different Hamiltonian or control bound."""
    expected={'Delta':float(Delta),'E':float(E),'T':float(T),'spin_scale':.5}
    for key,value in expected.items():
        actual=saved.get(key)
        if actual is None or not math.isclose(float(actual),value,rel_tol=1e-13,abs_tol=1e-14):
            raise ValueError(f'Checkpoint parameter mismatch: saved {key}={actual}, requested {value}. '
                'Resume requires the same physical parameters. Omit --resume to calculate the new case.')
    if saved.get('Hamiltonian')!='((Delta+ux)*sigma_x+uy*sigma_y+(theta+uz)*sigma_z)/2':
        raise ValueError('Checkpoint Hamiltonian convention does not match S_a=sigma_a/2.')


def residual(u,g,T,E):
    dt=T/len(u)
    return float(np.sqrt(dt)*np.linalg.norm(project(u+g/dt,E)-u))


def reference_objective(u,Delta,T,gradient=False):
    """Original Julia block-exponential algorithm with all spin factors corrected."""
    N=len(u);dt=T/N;states=np.zeros((N+1,4),complex);states[0,:2]=PLUS
    maps=[];matrices=[]
    for k,v in enumerate(u):
        H=((Delta+v[0])*SX+v[1]*SY+v[2]*SZ)/2
        L=np.zeros((4,4),complex);L[:2,:2]=H;L[2:,2:]=H;L[2:,:2]=SZ/2
        M=-1j*dt*L;P=expm(M);maps.append(P);matrices.append(M);states[k+1]=P@states[k]
    p,f=states[-1,:2],states[-1,2:];c=np.vdot(p,f)
    F=4*(np.vdot(f,f).real-abs(c)**2)
    if not gradient:return float(F),states
    lam=np.r_[-4*c.conjugate()*f,4*(f-c*p)];g=np.zeros_like(u)
    for k in range(N-1,-1,-1):
        for a in range(3):
            B=np.zeros((4,4),complex);B[:2,:2]=B[2:,2:]=-1j*dt*OPS[a]/2
            D=expm_frechet(matrices[k],B,compute_expm=False)
            g[k,a]=2*np.vdot(lam,D@states[k]).real
        lam=maps[k].conj().T@lam
    return float(F),g


def core_checks():
    rng=np.random.default_rng(1703);u=project(.15*rng.normal(size=(8,3)),.5)
    T=2.3;F,g=value_gradient(u,1.,T);reference,rg=reference_objective(u,1.,T,True)
    _,states=reference_objective(u,1.,T)
    directions=[]
    for _ in range(5):
        d=rng.normal(size=u.shape);d/=np.linalg.norm(d);eps=2e-6
        fd=(value_gradient(u+eps*d,1.,T,False)[0]-value_gradient(u-eps*d,1.,T,False)[0])/(2*eps)
        directions.append(abs(fd-np.sum(g*d)))
    theta_eps=2e-5
    plus=forward(u,1.,T,theta_eps)[-1,:2];minus=forward(u,1.,T,-theta_eps)[-1,:2]
    sensitivity_error=float(max(abs((plus-minus)/(2*theta_eps)-states[-1,2:])))
    analytic_errors=[]
    for t in (.2,4.,40.):
        f=value_gradient(np.zeros((30,3)),1.,t,False)[0]
        analytic_errors.append(abs(f-4*np.sin(t/2)**2))
    out={'QFI_vs_block_exponential_error':abs(F-reference),
        'state_vs_block_exponential_error':float(max(abs(forward(u,1.,T).ravel()-states.ravel()))),
        'gradient_vs_block_Frechet_max_error':float(np.max(abs(g-rg))),
        'gradient_directional_FD_max_error':float(max(directions)),
        'parameter_sensitivity_FD_max_error':sensitivity_error,
        'no_control_analytic_QFI_max_error':float(max(analytic_errors)),
        'spin_scale':.5,'no_control_QFI_formula':'4 sin^2(Delta*t/2)/Delta^2',
        'universal_QFI_bound':'T^2', 'passed':True}
    assert max(out[k] for k in out if k.endswith('error'))<2e-7,out
    print('Independent numerical checks:',json.dumps(out),flush=True)
    return out


def constant_qfi(u,Delta,T):
    h=np.array([Delta+u[0],u[1],u[2]]);w=np.linalg.norm(h)
    if w<1e-13:return T*T
    n=h/w
    # g = integral_0^T R_n(-w*t) e_z dt; h has no hidden factor of two.
    S=T*np.sinc(w*T/np.pi);C=.5*w*T*T*np.sinc(w*T/(2*np.pi))**2
    ez=np.array([0.,0.,1.]);g=T*n[2]*n+S*(ez-n[2]*n)-C*np.cross(n,ez)
    return float(g[1]*g[1]+g[2]*g[2])


def constant_best(Delta,E,T,rng):
    def radial(v):return E*np.asarray(v)/max(1.,np.linalg.norm(v))
    seeds=[np.zeros(3)]
    for a in range(3):
        for sign in (-1.,1.):
            v=np.zeros(3);v[a]=sign;seeds.append(v)
    for _ in range(32):
        v=rng.normal(size=3);v/=np.linalg.norm(v);seeds.append(v)
    seeds.append(np.array([-E/Delta,0.,np.sqrt(1-(E/Delta)**2)]))
    best=np.zeros(3);F=constant_qfi(best,Delta,T)
    for seed in seeds:
        result=minimize(lambda v:-constant_qfi(radial(v),Delta,T),seed,method='Nelder-Mead',
            options={'maxiter':1500,'xatol':1e-9,'fatol':1e-12})
        for candidate in (radial(seed),radial(result.x)):
            value=constant_qfi(candidate,Delta,T)
            if value>F:best,F=candidate.copy(),value
    return best,F


def solve(initial,Delta,E,T,iterations,tol,label):
    u=project(initial,E);dt=T/len(u);step=1.;history=[];started=time.perf_counter();last=started
    old_u=None;old_G=None;status='iteration limit'
    for it in range(iterations):
        F,g=value_gradient(u,Delta,T);G=g/dt;res=residual(u,g,T,E)
        history.append([it,F,res,time.perf_counter()-started])
        if res<tol:status='projected stationary point';break
        if old_u is not None:
            s=(u-old_u).ravel();y=(old_G-G).ravel();sy=float(s@y)
            if sy>1e-18:step=float(np.clip((s@s)/sy,1e-5,100.))
        eta=min(step,100.);accepted=False
        for trial in range(40):
            v=project(u+eta*G,E);change=v-u
            fv=value_gradient(v,Delta,T,False)[0]
            if fv>=F+1e-4*float(np.sum(g*change))-1e-12:
                old_u,old_G=u.copy(),G.copy();u=v;step=min(1.5*eta,100.);accepted=True;break
            eta*=.5
        if not accepted:status='line search stalled';break
        if time.perf_counter()-last>20:
            print(f'{label}: iter={it+1}, QFI={fv:.10f}, residual={res:.3g}',flush=True);last=time.perf_counter()
    F,g=value_gradient(u,Delta,T);res=residual(u,g,T,E)
    report={'label':label,'QFI':float(F),'projected_residual':res,'stationarity_passed':res<tol,
        'tolerance':tol,'iterations':it+1,'status':status,'intervals':len(u),
        'wall_seconds':time.perf_counter()-started}
    print(f'{label}: QFI={F:.10f}, residual={res:.3g}, {status}',flush=True)
    return u,report,np.asarray(history)


def profile(u,Delta,T,points=2001):
    N=len(u);edges=np.linspace(0,T,N+1)
    times=np.unique(np.r_[np.linspace(0,T,points),edges])
    states=forward(u,Delta,T)
    indices=np.clip(np.searchsorted(edges,times,side='right')-1,0,N-1)
    out=np.zeros((len(times),11));out[:,0]=times
    max_norm_error=0.;max_orthogonality=0.
    for j,(t,k) in enumerate(zip(times,indices)):
        h=np.array([Delta+u[k,0],u[k,1],u[k,2]])
        c=coefficients(h,t-edges[k]);p=states[k,:2];f=states[k,2:]
        pp=np.asarray(apply_u(p,h,c));ff=np.asarray(apply_u(f,h,c))+np.asarray(apply_du(p,2,h,c))
        scale=np.linalg.norm(pp);max_norm_error=max(max_norm_error,abs(scale-1.))
        pp/=scale;ff/=scale
        overlap=np.vdot(pp,ff);tangent=ff-pp*overlap
        r=np.array([np.vdot(pp,op@pp).real for op in OPS])
        source=-.5j*(SZ@pp-r[2]*pp);speed=np.linalg.norm(source);n=np.linalg.norm(tangent)
        complex_overlap=np.vdot(tangent,source)
        valid=n>1e-11 and speed>1e-11
        cosine=float(np.clip(complex_overlap.real/(n*speed),-1.,1.)) if valid else np.nan
        sine=float(np.clip(complex_overlap.imag/(n*speed),-1.,1.)) if valid else np.nan
        phase=float(np.angle(complex_overlap)) if valid else np.nan
        rate=2*complex_overlap.real/n if n>1e-11 else np.nan
        out[j,1:]=np.r_[4*n*n,r,speed,cosine,rate,8*complex_overlap.real,sine,phase]
        max_orthogonality=max(max_orthogonality,abs(np.vdot(pp,tangent)))
    return out,{'wavefunction_norm_max_error':max_norm_error,
        'pure_state_tangent_orthogonality_max_error':float(max_orthogonality),
        'Bloch_sphere_norm_max_error':float(max(abs(np.linalg.norm(out[:,2:5],axis=1)-1.)))}


def run_case(Delta,E,T,args):
    folder=OUT/f'T_{T:g}';folder.mkdir(exist_ok=True)
    pending=None
    if args.resume and (folder/'result.json').exists():
        saved=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        validate_saved_case(saved,Delta,E,T)
        if saved.get('stationarity_passed') and saved.get('grid_converged'):
            print(f'T={T:g}: reuse validated saved result.',flush=True);return saved
        pending=saved
    rng=np.random.default_rng(args.seed+round(100*T));uc,Fc=constant_best(Delta,E,T,rng)
    reference,unused=reference_objective(uc[None,:],Delta,T)
    assert abs(reference-Fc)<1e-8*max(1.,Fc)
    N=args.intervals;initials=[];labels=[]
    if pending:
        with np.load(folder/'profiles.npz') as old:
            u=old['u'].copy();uc=old['constant_u'].copy()
        Fc=pending['constant_QFI']
        grid=pending['grid_history'].copy();selected=grid[0].copy()
        selected['initial_source']=pending['selected_start_source']
        grid_converged=False
        print(f'T={T:g}: continue grid checks from {len(u)} saved intervals.',flush=True)
    cancel=np.zeros((N,3));cancel[:,0]=-E
    initials.extend([cancel,np.repeat(uc[None,:],N,axis=0),np.zeros((N,3))])
    labels.extend(['maximal drift cancellation','best constant from THIS run','zero control'])
    tau=(np.arange(N)+.5)/N
    for k in range(max(0,args.starts-3)):
        modes=np.array([np.ones(N),np.sin(2*np.pi*tau),np.cos(2*np.pi*tau),np.sin(4*np.pi*tau),np.cos(4*np.pi*tau)])
        initials.append(project((rng.normal(size=(3,5))@modes).T*E/3,E));labels.append(f'fresh random Fourier {k+1}')
    candidates=[]
    for k,initial in enumerate(initials[:args.starts] if pending is None else []):
        u,report,hist=solve(initial,Delta,E,T,args.max_iterations,args.tolerance,f'T={T:g}, start {k+1}/{args.starts}')
        report['initial_source']=labels[k]
        np.savez_compressed(folder/f'start_{k:02d}.npz',u=u,initial=initial,history=hist)
        dump(folder/f'start_{k:02d}.json',report);candidates.append((u,report))
    if pending is None:
        u,selected=max(candidates,key=lambda v:v[1]['QFI'])
        grid=[dict(selected)];grid_converged=False
    for level in range(len(grid),args.refinements+1):
        initial=np.repeat(u,2,axis=0)
        u,report,hist=solve(initial,Delta,E,T,args.max_iterations,args.tolerance,f'T={T:g}, grid x{2**level}')
        change=report['QFI']-grid[-1]['QFI'];threshold=max(1e-5,1e-5*report['QFI'])
        report.update(grid_gain=change,grid_tolerance=threshold)
        grid.append(report)
        np.savez_compressed(folder/f'grid_{level}.npz',u=u,history=hist);dump(folder/f'grid_{level}.json',report)
        if abs(change)<threshold and report['stationarity_passed']:
            grid_converged=True;break
    optimized,physical=profile(u,Delta,T)
    constant,const_physical=profile(uc[None,:],Delta,T)
    free,free_physical=profile(np.zeros((1,3)),Delta,T)
    assert max(abs(free[:,1]-4*np.sin(Delta*free[:,0]/2)**2/Delta**2))<1e-9
    bound_error=float(max(np.linalg.norm(u,axis=1))-E)
    assert bound_error<1e-12
    assert optimized[-1,1]<=T*T+1e-7 and optimized[-1,1]>=Fc-1e-7
    assert abs(optimized[-1,1]-grid[-1]['QFI'])<1e-8*max(1.,grid[-1]['QFI'])
    header='time,QFI,mean_sigmax,mean_sigmay,mean_sigmaz,J_F,cos_beta,d_sqrtF_dt,dF_dt,sin_beta,beta_signed'
    for name,data in (('optimized',optimized),('constant',constant),('uncontrolled',free)):
        np.savetxt(folder/f'{name}_trajectory.csv',data,delimiter=',',header=header,comments='')
    edges=np.linspace(0,T,len(u)+1)
    np.savetxt(folder/'optimal_controls.csv',np.column_stack((edges[:-1],edges[1:],u)),delimiter=',',comments='',header='t_start,t_end,ux,uy,uz')
    np.savez_compressed(folder/'profiles.npz',optimized=optimized,constant=constant,uncontrolled=free,
        u=u,edges=edges,constant_u=uc)
    row={'Delta':Delta,'E':E,'T':T,'Delta_T':Delta*T,'spin_scale':.5,
        'optimized_QFI':float(optimized[-1,1]),'constant_QFI':Fc,'uncontrolled_QFI':float(free[-1,1]),
        'constant_control':uc.tolist(),'initial_intervals':N,'final_intervals':len(u),
        'starts':args.starts,'selected_start_source':selected['initial_source'],
        'stationarity_passed':grid[-1]['stationarity_passed'],'projected_residual':grid[-1]['projected_residual'],
        'grid_converged':grid_converged,'grid_history':grid,'physical_checks':physical,
        'maximum_control_norm':float(max(np.linalg.norm(u,axis=1))),
        'old_control_inputs':False,'noise_samples':0,'global_optimum_proven':False,
        'Hamiltonian':'((Delta+ux)*sigma_x+uy*sigma_y+(theta+uz)*sigma_z)/2',
        'phase_definition':'arg <phi_perp, -i(Sz-<Sz>)psi>; full signed phase, not arccos(cos_beta)',
        'no_control_spin_half_analytic_check_passed':True}
    dump(folder/'result.json',row)
    print(f'T={T:g} FINAL: F={row["optimized_QFI"]:.10f}, constant={Fc:.10f}, control norm={row["maximum_control_norm"]:.8f}, stationary={row["stationarity_passed"]}, grid={grid_converged}',flush=True)
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__,parents=[_output_parser])
    parser.add_argument('--Delta',type=float,default=1.)
    parser.add_argument('--E',type=float,default=.5)
    parser.add_argument('--times',type=float,nargs='+',default=[4.,40.])
    parser.add_argument('--intervals',type=int,default=200)
    parser.add_argument('--starts',type=int,default=8)
    parser.add_argument('--max-iterations',type=int,default=3000)
    parser.add_argument('--refinements',type=int,default=4)
    parser.add_argument('--tolerance',type=float,default=1e-6)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--replot',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--no-plot',action='store_true',help='Compute and save data; draw separately afterward.')
    args=parser.parse_args();started=time.perf_counter();stamp=datetime.now().astimezone().isoformat()
    sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    print('Python:',sys.executable,'\nVersion:',sys.version.replace('\n',' '),'\nWorking directory:',Path.cwd(),flush=True)
    print('Manuscript convention: S=sigma/2. Three-axis control with Euclidean norm bound. No noise.',flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.replot:
        from plot_transverse_field_spin_prl import plot_results
        data_dir = OUT if (OUT/'results.json').is_file() else ROOT/'data'
        plot_results(data_dir, OUT)
        return
    assert 0<args.E<args.Delta and all(t>0 for t in args.times)
    if min(args.intervals,args.starts,args.max_iterations)<1 or args.refinements<0 or args.tolerance<=0:
        raise ValueError('Intervals, starts, iterations and tolerance must be positive; refinements cannot be negative.')
    if args.resume:
        # Check every saved case before modifying any manifests or result files.
        for T in args.times:
            cached=OUT/f'T_{T:g}'/'result.json'
            if cached.exists():validate_saved_case(json.loads(cached.read_text(encoding='utf-8')),args.Delta,args.E,T)
    checks=core_checks();dump(OUT/'core_checks.json',checks)
    if args.check_only:return
    manifest={'started_at':stamp,'python':sys.executable,'python_version':sys.version,
        'working_directory':str(Path.cwd()),'libraries':{'numpy':np.__version__,'scipy':scipy.__version__},
        'options':{**vars(args),'output_dir':str(OUT)},
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'physics':{'Delta':args.Delta,'E':args.E,'theta':0.,'initial_state':'+x','spin_scale':.5,
        'control_constraint':'ux^2+uy^2+uz^2 <= E^2'},'noise_samples':0,'original_zip_had_saved_data':False}
    dump(OUT/'run_manifest.json',manifest)
    rows=[run_case(args.Delta,args.E,T,args) for T in args.times]
    dump(OUT/'results.json',rows)
    if not args.no_plot:
        from plot_transverse_field_spin_prl import plot_results
        plot_results(OUT, OUT)
    report={'started_at':stamp,'finished_at':datetime.now().astimezone().isoformat(),
        'process_wall_seconds':time.perf_counter()-started,'noise_samples':0,
        'core_checks_passed':checks['passed'],'all_stationarity_passed':all(r['stationarity_passed'] for r in rows),
        'all_control_grid_checks_passed':all(r['grid_converged'] for r in rows),'results':rows}
    dump(OUT/'run_summary.json',report)
    print('Completed:',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':
    main()
