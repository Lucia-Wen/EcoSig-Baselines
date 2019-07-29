from baselines.common.models import mlp, cnn_small


def atari():
    return dict(
        network = cnn_small(),
        timesteps_per_batch=512,
        max_kl=0.001,
        cg_iters=10,
        cg_damping=1e-3,
        gamma=0.98,
        lam=1.0,
        vf_iters=3,
        vf_stepsize=1e-4,
        entcoeff=0.00,
    )

def mujoco():
    return dict(
        network = mlp(num_hidden=64, num_layers=4),
        timesteps_per_batch=1024,
        max_kl=0.01,
        cg_iters=10,
        cg_damping=0.1,
        gamma=1,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=True,
    )


def lucia_env():
    return dict(
        network = mlp(num_hidden=64, num_layers=6),
        timesteps_per_batch=1024*2,
        max_kl=0.015,
        cg_iters=10,
        cg_damping=0.1,
        gamma=0.995,
        lam=0.98,
        vf_iters=5,
        vf_stepsize=1e-3,
        normalize_observations=False,
    )
