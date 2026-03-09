import torch as th


def sample_initial_conditions(
    num_samples,
    theta1_range_deg=(-180.0, 180.0),
    theta2_range_deg=(-180.0, 180.0),
    omega1_range=(-8.0, 8.0),
    omega2_range=(-8.0, 8.0),
    seed=0,
):
    gen = th.Generator(device="cpu")
    gen.manual_seed(seed)

    def _uniform(low, high):
        return low + (high - low) * th.rand(num_samples, generator=gen)

    theta1 = _uniform(*theta1_range_deg)
    theta2 = _uniform(*theta2_range_deg)
    omega1 = _uniform(*omega1_range)
    omega2 = _uniform(*omega2_range)

    return th.stack([theta1, omega1, theta2, omega2], dim=1).to(th.float32)
