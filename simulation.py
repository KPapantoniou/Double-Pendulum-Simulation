import numpy as np
import torch as th

from chaoslab.simulation import PendulumParams, integrate_batch, resolve_device, total_energy
from chaoslab.visualization import animate_pendulum, plot_simulation_diagnostics


def run_single_simulation(
    theta1_deg=10.0,
    omega1=0.0,
    theta2_deg=130.0,
    omega2=0.0,
    time_horizon=30.0,
    dt=0.01,
    params=None,
    device=None,
):
    if params is None:
        params = PendulumParams(m1=2.0, m2=2.0, l1=1.0, l2=1.0, g=9.81)

    device = resolve_device(device)
    steps = int(time_horizon / dt)
    initial = th.tensor([[theta1_deg, omega1, theta2_deg, omega2]], dtype=th.float32)

    t_th, traj_th = integrate_batch(
        initial_states=initial,
        params=params,
        dt=dt,
        steps=steps,
        device=device,
        initial_in_degrees=True,
    )

    traj = traj_th[:, 0, :].detach().cpu()
    t = t_th.detach().cpu().numpy()
    theta1 = traj[:, 0].numpy()
    omega1_arr = traj[:, 1].numpy()
    theta2 = traj[:, 2].numpy()
    omega2_arr = traj[:, 3].numpy()

    x1 = params.l1 * np.sin(theta1)
    y1 = -params.l1 * np.cos(theta1)
    x2 = x1 + params.l2 * np.sin(theta2)
    y2 = y1 - params.l2 * np.cos(theta2)

    energy = total_energy(traj_th[:, 0, :], params).detach().cpu().numpy()
    return t, theta1, omega1_arr, theta2, omega2_arr, x1, y1, x2, y2, energy, device


def visualize_single_simulation():
    t, theta1, omega1, theta2, omega2, x1, y1, x2, y2, energy, device = run_single_simulation()
    print(f"Simulation device: {device}")

    plot_simulation_diagnostics(
        t=t,
        theta1=theta1,
        theta2=theta2,
        omega1=omega1,
        energy=energy,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        show=False,
    )
    _, ani = animate_pendulum(t=t, x1=x1, y1=y1, x2=x2, y2=y2, show=True)
    return ani


if __name__ == "__main__":
    _ani = visualize_single_simulation()
