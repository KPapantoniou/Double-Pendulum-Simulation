import torch as th

from chaoslab.simulation import PendulumParams, total_energy


def raw_state_features(initial_states_deg):
    # [theta1_deg, omega1, theta2_deg, omega2]
    theta1 = th.deg2rad(initial_states_deg[:, 0])
    omega1 = initial_states_deg[:, 1]
    theta2 = th.deg2rad(initial_states_deg[:, 2])
    omega2 = initial_states_deg[:, 3]
    return th.stack([theta1, theta2, omega1, omega2], dim=1)


def trig_state_features(initial_states_deg):
    theta1 = th.deg2rad(initial_states_deg[:, 0])
    omega1 = initial_states_deg[:, 1]
    theta2 = th.deg2rad(initial_states_deg[:, 2])
    omega2 = initial_states_deg[:, 3]
    return th.stack(
        [th.sin(theta1), th.cos(theta1), th.sin(theta2), th.cos(theta2), omega1, omega2],
        dim=1,
    )


def energy_variation_feature(traj_states_rad, params: PendulumParams):
    # traj_states_rad shape: [T, B, 4]
    energy = total_energy(traj_states_rad, params)
    return energy.amax(dim=0) - energy.amin(dim=0)
