from double_pendulum_custom_gym_env.double_pendulum_env import *
from gym.envs.registration import register

# delete if it's registered
env_name = 'double-pendulum-custom-v0'
if env_name in gym.envs.registry.env_specs:
    del gym.envs.registry.env_specs[env_name]

register(
    id='double-pendulum-custom-v0',
    entry_point='double_pendulum_custom_gym_env:DoublePendulumEnv',
    kwargs={'render_sim': False, 'n_steps': 1000, 'm1': 1, 'm2': 1, 'L1': 100, 'L2': 100, 'mode': 'balance'}
)