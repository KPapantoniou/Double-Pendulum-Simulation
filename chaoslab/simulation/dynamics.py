from dataclasses import dataclass

import torch as th


@dataclass(frozen=True)
class PendulumParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81


def resolve_device(requested_device=None):
    if requested_device:
        if requested_device == "cuda" and not th.cuda.is_available():
            print("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested_device

    if th.cuda.is_available():
        return "cuda"
    if hasattr(th.backends, "mps") and th.backends.mps.is_available():
        return "mps"
    return "cpu"


def _params_to_tensor(params: PendulumParams, device, dtype=th.float32):
    return {
        "m1": th.tensor(params.m1, device=device, dtype=dtype),
        "m2": th.tensor(params.m2, device=device, dtype=dtype),
        "l1": th.tensor(params.l1, device=device, dtype=dtype),
        "l2": th.tensor(params.l2, device=device, dtype=dtype),
        "g": th.tensor(params.g, device=device, dtype=dtype),
    }


def double_pendulum_rhs(state, p):
    theta1, omega1, theta2, omega2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    delta = theta2 - theta1

    a11 = (p["m1"] + p["m2"]) * p["l1"]
    a12 = p["m2"] * p["l2"] * th.cos(delta)
    a21 = p["l1"] * th.cos(delta)
    a22 = p["l2"]

    b1 = p["m2"] * p["l2"] * omega2.square() * th.sin(delta) - (p["m1"] + p["m2"]) * p["g"] * th.sin(theta1)
    b2 = -p["l1"] * omega1.square() * th.sin(delta) - p["g"] * th.sin(theta2)

    det = a11 * a22 - a12 * a21
    alpha1 = (b1 * a22 - b2 * a12) / det
    alpha2 = (a11 * b2 - a21 * b1) / det

    return th.stack([omega1, alpha1, omega2, alpha2], dim=1)


def rk4_step(state, dt, p):
    k1 = double_pendulum_rhs(state, p)
    k2 = double_pendulum_rhs(state + 0.5 * dt * k1, p)
    k3 = double_pendulum_rhs(state + 0.5 * dt * k2, p)
    k4 = double_pendulum_rhs(state + dt * k3, p)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_batch(
    initial_states,
    params: PendulumParams,
    dt=0.01,
    steps=6000,
    device=None,
    initial_in_degrees=True,
    save_stride=1,
    dtype=th.float32,
):
    device = resolve_device(device)
    p = _params_to_tensor(params, device, dtype=dtype)

    y = initial_states.to(device=device, dtype=dtype).clone()
    if initial_in_degrees:
        y[:, 0] = th.deg2rad(y[:, 0])
        y[:, 2] = th.deg2rad(y[:, 2])

    save_steps = steps // save_stride + 1
    traj = th.empty((save_steps, y.shape[0], 4), device=device, dtype=dtype)
    times = th.arange(0, steps + 1, save_stride, device=device, dtype=dtype) * dt

    traj[0] = y
    write_idx = 1
    for step in range(1, steps + 1):
        y = rk4_step(y, dt, p)
        if step % save_stride == 0:
            traj[write_idx] = y
            write_idx += 1

    return times, traj


def total_energy(states, params: PendulumParams):
    theta1, omega1, theta2, omega2 = states[..., 0], states[..., 1], states[..., 2], states[..., 3]

    v1_sq = (params.l1 * omega1).square()
    v2_sq = (
        (params.l1 * omega1).square()
        + (params.l2 * omega2).square()
        + 2.0 * params.l1 * params.l2 * omega1 * omega2 * th.cos(theta2 - theta1)
    )

    kinetic = 0.5 * params.m1 * v1_sq + 0.5 * params.m2 * v2_sq
    potential = -params.m1 * params.g * params.l1 * th.cos(theta1) - params.m2 * params.g * (
        params.l1 * th.cos(theta1) + params.l2 * th.cos(theta2)
    )
    return kinetic + potential
