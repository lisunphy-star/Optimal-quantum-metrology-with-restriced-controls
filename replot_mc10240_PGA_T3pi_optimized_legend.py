"""Render Figure 2 from the saved Monte Carlo profiles and controls."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    args = parser.parse_args()

    import spin_sensor_mc10240_PGA_T3pi as figure

    profiles, controls = {}, {}
    for sigma in figure.NOISE_STRENGTHS:
        profile_file = args.data_dir / f'profiles_sigma_{sigma:.1f}.npz'
        control_file = args.data_dir / figure.case_key(sigma, 3.) / 'controls.npz'
        with figure.np.load(profile_file, allow_pickle=False) as saved:
            profiles[sigma] = {key: saved[key].copy() for key in saved.files}
        with figure.np.load(control_file, allow_pickle=False) as saved:
            controls[sigma] = {key: saved[key].copy() for key in saved.files}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure.configure_style(figure.plt)
    names, _ = figure.plot_time_evolution(
        profiles, controls, figure.E_OVER_OMEGA, args.output_dir,
        figure.np, figure.matplotlib, figure.plt, figure.Line2D)
    print('Figure 2:', ', '.join(str(args.output_dir / name) for name in names))


if __name__ == '__main__':
    main()
