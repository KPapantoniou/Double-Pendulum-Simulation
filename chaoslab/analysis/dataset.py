from dataclasses import dataclass

import numpy as np
import torch as th

from chaoslab.analysis.features import raw_state_features, trig_state_features
from chaoslab.analysis.sampling import sample_initial_conditions
from chaoslab.numerics import estimate_lyapunov_batch
from chaoslab.simulation import PendulumParams, resolve_device


@dataclass
class DatasetConfig:
    num_samples: int = 50000
    batch_size: int = 4096
    dt: float = 0.01
    horizon: float = 30.0
    epsilon: float = 1e-5
    renorm_interval: int = 5
    seed: int = 42
    theta1_range_deg: tuple = (-180.0, 180.0)
    theta2_range_deg: tuple = (-180.0, 180.0)
    omega1_range: tuple = (-8.0, 8.0)
    omega2_range: tuple = (-8.0, 8.0)


@th.inference_mode()
def generate_dataset(params: PendulumParams, cfg: DatasetConfig, device=None):
    device = resolve_device(device)
    steps = int(cfg.horizon / cfg.dt)

    initial_states = sample_initial_conditions(
        num_samples=cfg.num_samples,
        theta1_range_deg=cfg.theta1_range_deg,
        theta2_range_deg=cfg.theta2_range_deg,
        omega1_range=cfg.omega1_range,
        omega2_range=cfg.omega2_range,
        seed=cfg.seed,
    )

    raw = raw_state_features(initial_states)
    trig = trig_state_features(initial_states)

    lyap_chunks = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        stop = min(start + cfg.batch_size, cfg.num_samples)
        batch = initial_states[start:stop]
        lyap = estimate_lyapunov_batch(
            batch,
            params,
            dt=cfg.dt,
            steps=steps,
            epsilon=cfg.epsilon,
            renorm_interval=cfg.renorm_interval,
            device=device,
            initial_in_degrees=True,
        )
        lyap_chunks.append(lyap.detach().cpu())

    lyap = th.cat(lyap_chunks, dim=0)

    return {
        "initial_states_deg": initial_states.numpy(),
        "X_raw": raw.numpy(),
        "X_trig": trig.numpy(),
        "y": lyap.numpy().astype(np.float32),
    }
