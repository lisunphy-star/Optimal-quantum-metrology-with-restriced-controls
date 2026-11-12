# Optimal quantum metrology with restricted controls

Python code and numerical data for Figures 1–3 of *Optimal quantum metrology with restricted controls* by Li Sun, Haoze Wang, and Chong Chen. [arXiv:2611.07583v1](https://arxiv.org/abs/2611.07583v1)

| Figure | Directory | Calculation |
| --- | --- | --- |
| 1 | [figure1_transverse_field](figure1_transverse_field/) | Spin-1/2 transverse-field sensing at `Delta*T = 4, 40` with `Delta = 1` and `E = 0.5`. Time-dependent three-axis control is optimized by projected gradient ascent and compared with optimized constant control and free evolution. |
| 2 | [figure2_ac_field](figure2_ac_field/) | AC-field sensing at `omega*T = 3*pi` with `E/omega = 8`, `omega*tau_c = pi`, and `sigma_xi/omega = 0, 0.4, 0.8`. For each noisy case, projected gradient ascent uses 10,240 fixed OU trajectories, and QFI is evaluated on 10,240 independent trajectories shared by the three protocols. |
| 3 | [figure3_terminal_qfi](figure3_terminal_qfi/) | Terminal QFI at `omega*T = 2*pi, 4*pi, ..., 20*pi` for `sigma_xi/omega = 0.8` with the same AC-field parameters. The OU average uses a deterministic Hermite hierarchy. The optimizer uses L-BFGS-B for the amplitudes and segment durations of the bounded control. |

## Replot the figures

Use Python 3.13 and run the commands from the repository root. The pinned dependencies match the tested Python 3.13.12 environment.

```bash
python -m pip install -r requirements.txt
python figure1_transverse_field/plot_transverse_field_spin_prl.py
python figure2_ac_field/replot_mc10240_PGA_T3pi_optimized_legend.py
python figure3_terminal_qfi/replot_hermite20_prl_single_column.py
```

Each script reads its adjacent `data/` directory and writes PDF, PNG, SVG, and EPS files to `figures/`. The `--data-dir` and `--output-dir` arguments select other input and output locations. Figure 2 also exports the two sensitivity growth curves for each noise strength as CSV files.

## Run the numerical calculations

```bash
python figure1_transverse_field/transverse_field_spin_prl.py --output-dir figure1_transverse_field/outputs/recompute
python figure2_ac_field/spin_sensor_mc10240_PGA_T3pi.py --output-dir figure2_ac_field/outputs/recompute --workers 3 --fine-iterations 4000
python figure3_terminal_qfi/spin_sensor_hermite20_even_completed.py --output-dir figure3_terminal_qfi/outputs/recompute --workers 4 --iterations 2500 --restarts 8
```

The Figure 2 command uses the recorded noise and control seeds. Figure 3 starts from the selected Hermite controls in `data/initial_controls/`. Add `--fresh` to generate new starting controls for Figure 3. Use `--resume` with the same output directory to continue a run.

The bundled data reproduce the published curves. A new optimization may reach a different local solution. For a fresh Figure 3 calculation, use its replot script with `--data-dir figure3_terminal_qfi/outputs/recompute` to obtain the single-column layout after convergence.

## Saved data

- Figure 1 stores the state, QFI, and sensitivity geometry for the three protocols in each `T_*/profiles.npz`, along with the control amplitudes and interval edges.
- Figure 2 stores the time profiles, QFI standard errors, and control waveforms for each noise strength. The training and evaluation noise files store the seeds and propagation grids used to regenerate each ensemble. Control amplitudes in `controls.npz` are in units of `E`, and edges are in units of `omega*t`.
- Figure 3 stores the terminal-QFI table, the final controls at each sampled time, and the Hermite coefficients of the terminal Bloch vector and its parameter derivative. The control amplitudes and edges use the same units as Figure 2.

Each `data/run_parameters.json` records the parameters and source hash from the original run. The repository copies adapt paths and figure export, including the “Optimized” legend used in Figure 2 of the paper.
