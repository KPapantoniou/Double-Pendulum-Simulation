# Double Pendulum Chaos Lab

A research-oriented Python project for:

1. Batched double-pendulum simulation.
2. Finite-time Lyapunov estimation.
3. Chaos dataset generation over phase space.
4. ML prediction of chaos intensity.
5. Scientific visual diagnostics.

## Project Structure

- `chaoslab/simulation/`: dynamics integration, parameters, energy.
- `chaoslab/numerics/`: Lyapunov estimation with renormalization.
- `chaoslab/analysis/`: sampling, feature engineering, dataset building, metrics.
- `chaoslab/models/`: model benchmarking (RF, GB, MLP).
- `chaoslab/visualization/`: plots for phase maps, histograms, residuals.
- `run_chaos_study.py`: full pipeline entrypoint.
- `simulation.py`: lightweight single-trajectory demo using `chaoslab` 

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Full chaos study

```bash
python run_chaos_study.py
```

This generates:

- Lyapunov distribution
- Chaos phase diagram
- Phase-space coloring by Lyapunov
- Prediction vs truth + residual plots for all available models

Artifacts are written to `outputs/`.

### 2) Single simulation demo

```bash
python simulation.py
```

This runs one pendulum trajectory and plots:

- angle evolution
- phase portrait
- energy consistency
- start/end geometric configuration
- pendulum animation

## Configuration

Main experiment settings live in `run_chaos_study.py` (`ExperimentConfig`):

- `num_samples`
- `batch_size`
- `dt`
- `horizon`
- `epsilon`
- `renorm_interval`
- `phase_grid`

For larger studies (50k+ samples), increase `num_samples` and tune `batch_size` to your GPU/CPU memory limits.

## Notes

- If `scikit-learn` is unavailable, RF/GB are skipped and MLP still runs.
- Device selection is automatic (`cuda` -> `mps` -> `cpu`).
- Lyapunov estimation uses angle-wrapped distance and renormalization for better numerical stability.
