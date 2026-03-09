import torch as th

from chaoslab.simulation import PendulumParams, integrate_batch, resolve_device, rk4_step
from chaoslab.simulation.dynamics import _params_to_tensor


def _wrap_angle_delta(delta):
    return th.atan2(th.sin(delta), th.cos(delta))


def _state_delta(a, b):
    d = a - b
    d[:, 0] = _wrap_angle_delta(d[:, 0])
    d[:, 2] = _wrap_angle_delta(d[:, 2])
    return d


def estimate_lyapunov_batch(
    initial_states,
    params: PendulumParams,
    dt=0.01,
    steps=6000,
    epsilon=1e-7,
    renorm_interval=10,
    device=None,
    initial_in_degrees=True,
):
    """
    Benettin-style finite-time Lyapunov estimate with periodic renormalization.

    Returns:
        lyapunov: tensor [batch]
    """
    device = resolve_device(device)
    dtype = th.float64
    p = _params_to_tensor(params, device, dtype=dtype)

    y = initial_states.to(device=device, dtype=dtype).clone()
    if initial_in_degrees:
        y[:, 0] = th.deg2rad(y[:, 0])
        y[:, 2] = th.deg2rad(y[:, 2])

    direction = th.randn_like(y)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    y_pert = y + epsilon * direction

    sum_logs = th.zeros(y.shape[0], device=device, dtype=dtype)
    total_time = 0.0
    clamp_floor = th.tensor(1e-24, device=device, dtype=dtype)

    full_chunks = steps // renorm_interval
    remainder = steps % renorm_interval

    for _ in range(full_chunks):
        for _ in range(renorm_interval):
            y = rk4_step(y, dt, p)
            y_pert = rk4_step(y_pert, dt, p)

        delta = _state_delta(y_pert, y)
        dist = delta.norm(dim=1).clamp_min(clamp_floor)
        sum_logs += th.log(dist / epsilon)

        delta = delta / dist.unsqueeze(1) * epsilon
        y_pert = y + delta
        total_time += renorm_interval * dt

    if remainder > 0:
        for _ in range(remainder):
            y = rk4_step(y, dt, p)
            y_pert = rk4_step(y_pert, dt, p)

        delta = _state_delta(y_pert, y)
        dist = delta.norm(dim=1).clamp_min(clamp_floor)
        sum_logs += th.log(dist / epsilon)
        total_time += remainder * dt

    return (sum_logs / max(total_time, 1e-12)).to(th.float32)


def simulate_pairwise_divergence(
    initial_states,
    params: PendulumParams,
    dt=0.01,
    steps=6000,
    epsilon=1e-7,
    device=None,
    initial_in_degrees=True,
):
    """Simulate base and perturbed trajectories for diagnostics/plots."""
    device = resolve_device(device)

    y = initial_states.to(device=device, dtype=th.float32).clone()
    if initial_in_degrees:
        y[:, 0] = th.deg2rad(y[:, 0])
        y[:, 2] = th.deg2rad(y[:, 2])

    direction = th.randn_like(y)
    direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
    y_pert = y + epsilon * direction

    _, traj = integrate_batch(y, params, dt=dt, steps=steps, device=device, initial_in_degrees=False)
    _, traj_pert = integrate_batch(y_pert, params, dt=dt, steps=steps, device=device, initial_in_degrees=False)
    return traj, traj_pert
