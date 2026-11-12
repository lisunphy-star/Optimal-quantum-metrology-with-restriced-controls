"""Render Figure 3 from the saved Hermite terminal-QFI results."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
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

STEM = 'qfi_hermite_x_only_20pi_completed'
COLORS = {'optimized': '#1689D4', 'cpmg': '#E58000'}
STYLE = {
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 10,
    'axes.labelsize': 11.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5, 'axes.linewidth': .8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 3.2, 'ytick.major.size': 3.2,
    'xtick.minor.size': 1.9, 'xtick.major.width': .75, 'ytick.major.width': .75,
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.bbox': None, 'savefig.pad_inches': 0,
}

def plot_results(data_dir, output_dir):
    rows = json.loads((data_dir/'results.json').read_text(encoding='utf-8'))
    verification = json.loads((data_dir/'verification.json').read_text(encoding='utf-8'))
    assert verification['passed'] and verification['all_optimization_convergence_met']
    assert verification['all_mesh_checks_passed'] and verification['all_time_checks_passed']
    assert not verification['Monte_Carlo_used'] and verification['noise_samples'] == 0
    assert [r['T_over_pi'] for r in rows] == list(range(2, 21, 2))
    assert all(r['optimization_convergence_met'] and not r['finite_budget_fast_mode'] for r in rows)
    x = np.r_[0., [r['omega_T'] for r in rows]]
    optimized = np.r_[0., [r['variational_QFI'] for r in rows]]
    cpmg = np.r_[0., [r['CPMG_QFI'] for r in rows]]
    plt.rcdefaults()
    plt.rcParams.update(STYLE)
    # Export the complete canvas, preserving the exact 3.375-inch column width.
    fig, ax = plt.subplots(figsize=(3.375, 2.5))
    fig.subplots_adjust(left=.175, right=.965, bottom=.185, top=.97)
    ax.plot(x, optimized, color=COLORS['optimized'], ls='-', lw=1.25,
            marker='o', ms=4.2, mew=.85, mfc=COLORS['optimized'], zorder=3)
    ax.plot(x, cpmg, color=COLORS['cpmg'], ls='--', lw=1.2,
            marker='s', ms=3.9, mew=.9, mfc='white', zorder=2)
    ax.set_xlim(-.30*np.pi, 20.40*np.pi)
    ax.set_ylim(-.45, 24.)
    major = np.arange(0, 21, 4)
    ax.set_xticks(major*np.pi, [r'$0$' if n == 0 else rf'${n}\pi$' for n in major])
    ax.set_xticks(np.arange(2, 20, 4)*np.pi, minor=True)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_xlabel(r'$\omega T$', labelpad=3.)
    ax.set_ylabel(r'$F_\theta$', labelpad=3.)
    handles = [
        Line2D([], [], color=COLORS['optimized'], ls='-', lw=1.25,
               marker='o', ms=4.2, label='Variational'),
        Line2D([], [], color=COLORS['cpmg'], ls='--', lw=1.2,
               marker='s', ms=3.9, mfc='white', mew=.9, label='CPMG'),
    ]
    legend = ax.legend(handles=handles, loc='upper left', frameon=False,
                       borderaxespad=.45, handlelength=2.05,
                       labelspacing=.32, handletextpad=.65)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    assert axes_box.contains(legend_box.x0, legend_box.y0)
    assert axes_box.contains(legend_box.x1, legend_box.y1)
    # Check the enlarged labels and ticks fit the physical page without cropping.
    for artist in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]:
        box = artist.get_window_extent(renderer)
        assert box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.bbox.width and box.y1 <= fig.bbox.height
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png', 'svg', 'eps'):
        path = output_dir/f'{STEM}_PRL_single_column.{ext}'
        fig.savefig(path, dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(f'Figure 3: {output_dir / (STEM + "_PRL_single_column.pdf")}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    args = parser.parse_args()
    plot_results(args.data_dir, args.output_dir)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    main()
