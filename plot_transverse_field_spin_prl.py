"""Render Figure 1 from the saved spin-half transverse-field data."""
from pathlib import Path
import argparse
import json
import os
import sys
ROOT=Path(__file__).resolve().parent
if os.name == 'nt':
    environment = Path(sys.executable).resolve().parent
    os.environ['PATH'] = os.pathsep.join(str(p) for p in
        (environment, environment/'Library'/'bin', environment/'Scripts')) + os.pathsep + os.environ.get('PATH', '')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'outputs'/'matplotlib_config'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

COLORS={'optimized':'#1689D4','constant':'#E58000','uncontrolled':'#777777'}
CONTROL_COLORS={'ux':'#2A8F55','uy':'#E58000','uz':'#8054A4','norm':'black'}
STYLES={'optimized':'-','constant':'--','uncontrolled':'-.'}
LABELS={'optimized':'Optimized','constant':'Constant control','uncontrolled':'No control'}
STYLE={'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':8,
    'axes.labelsize':9,'axes.titlesize':9,'xtick.labelsize':7,'ytick.labelsize':7,
    'axes.linewidth':.65,'xtick.direction':'in','ytick.direction':'in',
    'xtick.top':True,'ytick.right':True,'xtick.major.size':2.5,'ytick.major.size':2.5,
    'text.color':'black','axes.labelcolor':'black','axes.edgecolor':'black',
    'xtick.color':'black','ytick.color':'black','figure.facecolor':'white','axes.facecolor':'white',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'}


def project_bloch(r):
    az,el=np.deg2rad([35.,25.])
    x,y,z=np.asarray(r).T
    return -np.sin(az)*x+np.cos(az)*y,-np.sin(el)*np.cos(az)*x-np.sin(el)*np.sin(az)*y+np.cos(el)*z


def sphere(ax,profiles,constant_period=None):
    theta=np.linspace(0,2*np.pi,241)
    ax.plot(np.cos(theta),np.sin(theta),color='.78',lw=.45,zorder=0)
    for lat in np.linspace(-np.pi/2,np.pi/2,9):
        r=np.column_stack((np.cos(lat)*np.cos(theta),np.cos(lat)*np.sin(theta),np.full_like(theta,np.sin(lat))))
        ax.plot(*project_bloch(r),color='.88',lw=.35,zorder=0)
    for lon in np.linspace(0,np.pi,10):
        r=np.column_stack((np.cos(lon)*np.cos(theta),np.sin(lon)*np.cos(theta),np.sin(theta)))
        ax.plot(*project_bloch(r),color='.88',lw=.35,zorder=0)
    for a,label in enumerate(('x','y','z')):
        e=np.eye(3)[a]
        x,y=project_bloch(np.array([-1.12*e,1.12*e]))
        ax.annotate('',xy=(x[1],y[1]),xytext=(x[0],y[0]),
            arrowprops={'arrowstyle':'->','color':'.45','lw':.45,'mutation_scale':5},zorder=1)
        x,y=project_bloch(1.27*e)
        ax.text(x,y,f'${label}$',ha='center',va='center',fontsize=8)
    for name in ('constant','optimized'):
        data=profiles[name];orbit=data[:,2:5]
        if name=='constant' and constant_period is not None and data[-1,0]>constant_period:
            # Repeated passes fill each other's dash gaps. The geometric orbit
            # is exactly periodic; show its first cycle with exact closure.
            orbit=np.vstack((data[data[:,0]<constant_period,2:5],data[0,2:5]))
        x,y=project_bloch(orbit)
        style=(0,(3.2,2.2)) if name=='constant' else STYLES[name]
        ax.plot(x,y,color=COLORS[name],ls=style,lw=.85,dash_capstyle='butt',zorder=3)
        # Keep the physical endpoint at T, not the endpoint of the display cycle.
        xt,yt=project_bloch(data[-1,2:5])
        ax.plot(xt,yt,'o',ms=2.5,mec='black',mew=.35,color=COLORS[name],zorder=4)
    x,y=project_bloch(np.array([1.,0.,0.]))
    ax.plot(x,y,'o',mfc='white',mec='.35',ms=3.2,mew=.6,zorder=5)
    ax.set(xlim=(-1.36,1.36),ylim=(-1.36,1.36),aspect='equal')
    ax.axis('off')


def polar(ax,profiles):
    angles=np.linspace(0,2*np.pi,361)
    for radius in (.125,.25,.375,.5):
        ax.plot(radius*np.cos(angles),radius*np.sin(angles),color='.84',lw=.45,zorder=0)
        ax.text(radius*np.cos(np.pi/8),radius*np.sin(np.pi/8),f'{radius:.2g}',
            color='.45',fontsize=5.5,ha='center',va='center',
            bbox={'facecolor':'white','edgecolor':'none','pad':.1})
    labels=(r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$',r'$5\pi/4$',r'$3\pi/2$',r'$7\pi/4$')
    for angle,label in zip(np.arange(8)*np.pi/4,labels):
        ax.plot([0,.5*np.cos(angle)],[0,.5*np.sin(angle)],color='.86',lw=.4,zorder=0)
        ax.text(.568*np.cos(angle),.568*np.sin(angle),label,ha='center',va='center',fontsize=7)
    for name in ('constant','optimized'):
        data=profiles[name];x=data[:,5]*data[:,6];y=data[:,5]*data[:,9]
        ax.plot(x,y,color=COLORS[name],ls=STYLES[name],lw=.85,zorder=3)
        # A few small arrows at separated times replace the old clustered heads.
        for fraction in (.25,.55,.85):
            k=int(np.argmin(abs(data[:,0]-fraction*data[-1,0])));j=max(0,k-3)
            if np.all(np.isfinite([x[j],y[j],x[k],y[k]])) and np.hypot(x[k]-x[j],y[k]-y[j])>1e-6:
                ax.annotate('',xy=(x[k],y[k]),xytext=(x[j],y[j]),
                    arrowprops={'arrowstyle':'->','color':COLORS[name],'lw':.65,'mutation_scale':4},zorder=4)
        if np.all(np.isfinite([x[-1],y[-1]])):
            ax.plot(x[-1],y[-1],'o',ms=2.6,mec='black',mew=.35,color=COLORS[name],zorder=5)
    ax.plot(.5,0.,'o',mfc='white',mec='.35',ms=3.2,mew=.6,zorder=6)
    ax.set(xlim=(-.62,.62),ylim=(-.62,.62),aspect='equal')
    ax.axis('off')


def plot_results(data_dir, output_dir):
    rows=json.loads((data_dir/'results.json').read_text(encoding='utf-8'))
    assert len(rows)==2 and all(r['spin_scale']==.5 for r in rows)
    plt.rcdefaults();plt.rcParams.update(STYLE)
    fig,axes=plt.subplots(2,4,figsize=(7.2,4.6))
    fig.subplots_adjust(left=.065,right=.985,bottom=.085,top=.845,wspace=.38,hspace=.52)
    for ax,title in zip(axes[0],('QFI','Control field','Bloch trajectory','Sensitivity geometry')):
        ax.set_title(title,pad=10)
    control_handles=[Line2D([],[],color=color,ls=style,lw=.85,label=label) for color,style,label in (
        (CONTROL_COLORS['ux'],'-',r'$u_x/E$'),(CONTROL_COLORS['uy'],'-',r'$u_y/E$'),
        (CONTROL_COLORS['uz'],'-',r'$u_z/E$'),(CONTROL_COLORS['norm'],'--',r'$|\mathbf{u}|/E$'))]
    for i,row in enumerate(rows):
        T,Delta,E=row['T'],row['Delta'],row['E']
        with np.load(data_dir/f'T_{T:g}'/'profiles.npz', allow_pickle=False) as d:
            profiles={n:d[n].copy() for n in ('optimized','constant','uncontrolled')}
            u,edges=d['u'].copy(),d['edges'].copy()
            constant_u=d['constant_u'].copy()
        ax=axes[i,0];maxF=0.
        for name in ('uncontrolled','constant','optimized'):
            data=profiles[name]
            ax.plot(Delta*data[:,0],data[:,1],color=COLORS[name],ls=STYLES[name],lw=.95,zorder=3)
            maxF=max(maxF,float(data[:,1].max()))
            if name!='uncontrolled':
                ax.plot(Delta*T,data[-1,1],'o',ms=2.8,mec='black',mew=.35,color=COLORS[name])
        ax.set_ylim(0,maxF*1.18);ax.set_ylabel(r'$F_\theta$')
        ax.text(.04,.93,rf'$\Delta T={Delta*T:g}$',transform=ax.transAxes,va='top',fontsize=8)
        ax=axes[i,1]
        for a,color,label in ((0,CONTROL_COLORS['ux'],r'$u_x/E$'),(1,CONTROL_COLORS['uy'],r'$u_y/E$'),(2,CONTROL_COLORS['uz'],r'$u_z/E$')):
            ax.stairs(u[:,a]/E,Delta*edges,baseline=None,color=color,lw=.85,label=label)
        ax.stairs(np.linalg.norm(u,axis=1)/E,Delta*edges,baseline=None,color='black',ls='--',lw=.75,label=r'$|\mathbf{u}|/E$')
        for bound in (-1.,1.):ax.axhline(bound,color='.65',ls=':',lw=.45,zorder=0)
        ax.set_ylim(-1.12,1.16);ax.set_yticks([-1,0,1]);ax.set_ylabel(r'$u_a/E$')
        # The positive middle region is clear of both control traces and the norm.
        ax.legend(handles=control_handles,loc='center',bbox_to_anchor=(.50,.70),
            ncol=2,frameon=False,fontsize=6.5,handlelength=1.6,columnspacing=.8,
            labelspacing=.25,borderpad=.15,handletextpad=.4)
        for ax in axes[i,:2]:
            ax.set_xlim(0,Delta*T)
            ticks=np.linspace(0,Delta*T,5)
            ax.set_xticks(ticks,[f'{v:g}' for v in ticks]);ax.set_xlabel(r'$\Delta t$')
            for spine in ax.spines.values():spine.set_color('black');spine.set_linewidth(.65)
        axes[i,0].set_xlim(0,1.025*Delta*T)
        constant_period=2*np.pi/np.linalg.norm(constant_u+np.array([Delta,0.,0.]))
        sphere(axes[i,2],profiles,constant_period);polar(axes[i,3],profiles)
        for j,ax in enumerate(axes[i]):
            ax.text(-.11,1.055,f'({chr(97+i)}{j+1})',transform=ax.transAxes,fontsize=9,color='black')
    handles=[Line2D([],[],color=COLORS[n],ls=STYLES[n],lw=1.1,label=LABELS[n]) for n in ('uncontrolled','optimized','constant')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,frameon=False,fontsize=8)
    stem='transverse_field_spin_half'+'_T'+'_T'.join(f'{r["Delta_T"]:g}' for r in rows)+'_PRL'
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf','svg','eps'):
        target=output_dir/f'{stem}.{ext}'
        fig.savefig(target,dpi=600,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    print(f'Figure 1: {output_dir / (stem + ".pdf")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    args = parser.parse_args()
    plot_results(args.data_dir, args.output_dir)


if __name__=='__main__':
    main()
