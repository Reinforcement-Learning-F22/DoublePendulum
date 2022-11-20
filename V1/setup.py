from setuptools import setup, find_packages
import os

DESCRIPTION = 'OpenAI Gym environment designed for training RL agents to balance double Pendulum.'
LONG_DESCRIPTION = ('This package contains OpenAI Gym environment designed for training RL agents to balance double Pendulum. '
                    'The environment is automatically registered under id: double-pendulum-custom-v0, '
                    'so it can be easily used by RL agent training libraries, such as StableBaselines3.<br /><br />At the '
                    'https://github.com/marek-robak/Double-cartpole-custom-gym-env-for-reinforcement-learning.git')

setup(
    name='double_pendulum_custom_gym_env',
    version='0',
    author='Ali Jnadi',
    author_email='alijenedie@gmail.com',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    # url='https://github.com/marek-robak/Double-cartpole-custom-gym-env-for-reinforcement-learning.git',
    # download_url='https://pypi.org/project/double-cartpole-custom-gym-env/',
    packages=find_packages(),
    include_package_data = True,
    install_requires=['gym', 'pygame', 'pymunk', 'numpy', 'stable-baselines3[extra]'],
    keywords=['reinforcement learning', 'gym environment', 'StableBaselines3', 'OpenAI Gym']
)
