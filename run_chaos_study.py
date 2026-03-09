from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chaoslab.analysis import DatasetConfig, generate_dataset
from chaoslab.models import benchmark_models
from chaoslab.simulation import PendulumParams, resolve_device
from chaoslab.visualization import (
    plot_chaos_phase_diagram,
    plot_lyapunov_histogram,
    plot_phase_space_by_lyapunov,
    plot_prediction_residuals,
)


@dataclass
class ExperimentConfig:
    outdir: str = "outputs"
    num_samples: int = 50000
    batch_size: int = 4096
    dt: float = 0.01
    horizon: float = 20.0
    epsilon: float = 1e-5
    renorm_interval: int = 5
    random_seed: int = 42
    make_phase_diagram: bool = True
    phase_grid: int = 60


def _print_metrics(title, metrics):
    print(f"\n[{title}]")
    for model_name, info in metrics.items():
        if "error" in info:
            print(f"- {model_name}: {info['error']}")
            continue
        print(f"- {model_name}: R2={info['r2']:.4f} MAE={info['mae']:.4f}")


def _print_target_diagnostics(y):
    p = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    vals = np.percentile(y, p)
    print("\n[Target diagnostics: finite-time Lyapunov]")
    print("Percentiles:")
    for pi, vi in zip(p, vals):
        print(f"- p{pi:>3}: {vi:.6f}")
    std = float(np.std(y))
    print(f"- std: {std:.6f}")
    if std < 1e-2:
        print("WARNING: Target variance is very low; models will collapse near the mean.")


def _best_available_model(metrics):
    valid = {k: v for k, v in metrics.items() if "error" not in v}
    if not valid:
        return None, None
    name = max(valid, key=lambda k: valid[k]["r2"])
    return name, valid[name]


def _plot_all_available_models(metrics, suffix, outdir):
    plotted = []
    for model_name, info in metrics.items():
        if "error" in info:
            print(f"- Skipping {model_name} ({suffix}): {info['error']}")
            continue
        plot_prediction_residuals(
            info["y_true"],
            info["y_pred"],
            model_name=f"{model_name}_{suffix}",
            outdir=outdir,
        )
        plotted.append(model_name)
    return plotted


def main():
    cfg = ExperimentConfig()
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(None)
    print(f"Running on device: {device}")

    params = PendulumParams(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    data_cfg = DatasetConfig(
        num_samples=cfg.num_samples,
        batch_size=cfg.batch_size,
        dt=cfg.dt,
        horizon=cfg.horizon,
        epsilon=cfg.epsilon,
        renorm_interval=cfg.renorm_interval,
        seed=cfg.random_seed,
    )

    data = generate_dataset(params=params, cfg=data_cfg, device=device)
    _print_target_diagnostics(data["y"])

    metrics_raw = benchmark_models(data["X_raw"], data["y"], seed=cfg.random_seed, device=device)
    metrics_trig = benchmark_models(data["X_trig"], data["y"], seed=cfg.random_seed, device=device)

    _print_metrics("Raw state features [theta1, theta2, omega1, omega2]", metrics_raw)
    _print_metrics("Trig state features [sin/cos + omega]", metrics_trig)

    plotted_raw = _plot_all_available_models(metrics_raw, "raw", cfg.outdir)
    plotted_trig = _plot_all_available_models(metrics_trig, "trig", cfg.outdir)

    best_name, best = _best_available_model(metrics_trig)
    if best is not None:
        print(f"Best trig model: {best_name}")
    if not plotted_raw and not plotted_trig:
        print("No model residual plots were generated (all models unavailable).")

    plot_lyapunov_histogram(data["y"], outdir=cfg.outdir)
    plot_phase_space_by_lyapunov(data["initial_states_deg"], data["y"], outdir=cfg.outdir)

    if cfg.make_phase_diagram:
        plot_chaos_phase_diagram(
            params,
            grid_size=cfg.phase_grid,
            dt=cfg.dt,
            horizon=cfg.horizon,
            epsilon=cfg.epsilon,
            renorm_interval=cfg.renorm_interval,
            device=device,
            outdir=cfg.outdir,
        )

    print(f"\nArtifacts saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
