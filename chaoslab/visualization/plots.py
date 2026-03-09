from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from chaoslab.numerics import estimate_lyapunov_batch
from chaoslab.simulation import PendulumParams, resolve_device


def _ensure_outdir(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def plot_phase_space_by_lyapunov(initial_states_deg, lyap, outdir="outputs"):
    outdir = _ensure_outdir(outdir)

    th1 = np.deg2rad(initial_states_deg[:, 0])
    om1 = initial_states_deg[:, 1]
    th2 = np.deg2rad(initial_states_deg[:, 2])
    om2 = initial_states_deg[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    sc1 = axes[0].scatter(th1, om1, c=lyap, s=8, cmap="plasma", alpha=0.75)
    axes[0].set_title("Phase Space: theta1 vs omega1")
    axes[0].set_xlabel("theta1 [rad]")
    axes[0].set_ylabel("omega1 [rad/s]")

    sc2 = axes[1].scatter(th2, om2, c=lyap, s=8, cmap="plasma", alpha=0.75)
    axes[1].set_title("Phase Space: theta2 vs omega2")
    axes[1].set_xlabel("theta2 [rad]")
    axes[1].set_ylabel("omega2 [rad/s]")

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist())
    cbar.set_label("Finite-time Lyapunov")
    fig.savefig(outdir / "phase_space_colored_by_lyapunov.png", dpi=160)
    plt.close(fig)


def plot_lyapunov_histogram(lyap, outdir="outputs"):
    outdir = _ensure_outdir(outdir)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lyap, bins=60, color="tab:blue", alpha=0.85)
    ax.set_xlabel("Finite-time Lyapunov")
    ax.set_ylabel("Count")
    ax.set_title("Lyapunov Distribution")
    fig.savefig(outdir / "lyapunov_histogram.png", dpi=160)
    plt.close(fig)


def plot_prediction_residuals(y_true, y_pred, model_name, outdir="outputs"):
    outdir = _ensure_outdir(outdir)

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(y_true, y_pred, s=10, alpha=0.6)
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=1)
    axes[0].set_title(f"{model_name}: prediction vs truth")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")

    axes[1].scatter(y_pred, residuals, s=10, alpha=0.6)
    axes[1].axhline(0.0, color="r", linestyle="--", linewidth=1)
    axes[1].set_title(f"{model_name}: residuals")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")

    fig.savefig(outdir / f"prediction_residuals_{model_name.lower()}.png", dpi=160)
    plt.close(fig)


@th.inference_mode()
def plot_chaos_phase_diagram(
    params: PendulumParams,
    theta1_range=(-180.0, 180.0),
    theta2_range=(-180.0, 180.0),
    omega1=0.0,
    omega2=0.0,
    grid_size=60,
    dt=0.01,
    horizon=20.0,
    epsilon=1e-7,
    renorm_interval=10,
    device=None,
    outdir="outputs",
):
    outdir = _ensure_outdir(outdir)
    device = resolve_device(device)

    th1 = np.linspace(theta1_range[0], theta1_range[1], grid_size)
    th2 = np.linspace(theta2_range[0], theta2_range[1], grid_size)
    g1, g2 = np.meshgrid(th1, th2)

    init = np.stack(
        [g1.ravel(), np.full(g1.size, omega1), g2.ravel(), np.full(g2.size, omega2)],
        axis=1,
    )
    init_t = th.tensor(init, dtype=th.float32)

    steps = int(horizon / dt)
    lyap = estimate_lyapunov_batch(
        init_t,
        params,
        dt=dt,
        steps=steps,
        epsilon=epsilon,
        renorm_interval=renorm_interval,
        device=device,
        initial_in_degrees=True,
    )

    img = lyap.detach().cpu().numpy().reshape(grid_size, grid_size)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        img,
        origin="lower",
        extent=[theta1_range[0], theta1_range[1], theta2_range[0], theta2_range[1]],
        aspect="auto",
        cmap="inferno",
    )
    ax.set_title("Chaos Phase Diagram (omega1=omega2=0)")
    ax.set_xlabel("theta1 [deg]")
    ax.set_ylabel("theta2 [deg]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Finite-time Lyapunov")
    fig.savefig(outdir / "chaos_phase_diagram.png", dpi=170)
    plt.close(fig)
