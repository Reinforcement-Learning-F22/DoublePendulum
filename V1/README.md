# Double-Pendulum-custom-gym-env-for-reinforcement-learning

This repository contains OpenAI Gym environment designed for teaching RL
agents the ability to bring the double pendulum upright and its further balancing.
To make this easy to use, the environment has been packed into a Python package,
which automatically registers the environment in the Gym library when the package
is included in the code. As a result, it can be easily used in conjunction with
reinforcement learning libraries such as StableBaselines3. There is also a sample
code for training and evaluating agents in this environment.

<p align="center">
  <img src="https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/Double%20Pendulum%20Swing%20up.gif"/>
</p>

## Installation

These instructions will guide you through installation of the environment and
show you how to use it.

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. Navigate to repository folder
1. 
```
python3 -m pip install -e .
```

### How to use it in your code

Now all you need to do to use this environment in your code is import the package.
After that, you can use it with Gym and StableBaselines3 library via its
id: double-pendulum-custom-v0.

```
from stable_baselines3 import PPO
import gym

import double_pendulum_custom_gym_env

model_path = './models'
log_path = './logs'

env = gym.make('double-pendulum-custom-v0', render_sim=False, n_steps=1000, mode = 'swing_up')

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=1500000)

PPO_path = './models/new_agent_swing_up'
model.save(PPO_path)
```

### Environment prerequisites

Environment to run needs Python3 with Gym, Pygame, Pymunk, Numpy and StableBaselines3
libraries. All of them are automatically installed when the package is installed.

## Environment details

This environment consists of a double pendulum. The RL agent can control the tourq acting
on the first pendulum. This tourq influences the movement of the first pendulum, which then translates
into the swing angle of the second pendulum. The agent's goal is to learn how to place the
second pendulum vertically and then the first one.

The physics engine for this environment runs at 60fps.

### Initial episode conditions

At the beginning of each episode, the double pendulum is placed that both the first and the second pedulum are pointing downward.

### Ending episode conditions

Each episode ends if the set number of timesteps
has passed.

### Agent action and observation space

The space of actions made available to the RL agent consists of one value from -1
to 1. It is correlated with the tourq acting on the first pendulum, -1 and 1 are maximum tourqs,
0 is no force acting.

The observation space consists of fours values, all ranging from -1 to 1.
- The first and second number informs about the current swing angle of the pendulum.
It is scaled to return 0 for a vertically upward pendulum, and -1 and 1 for a
downward pendulum.
- The third and fourth number contains information about the angular velocity of the pendulum.
It has been scaled so that the values -1 and 1 represent the maximum achievable values.

### Reward function

The task that the agent must perform consists of two phases. In the first one, it has to swing the second pendulum vertically. 
In the second, while keeping the balance, it has to move the first pendulum to the target point. 
For this reason, the reward function has been split into two expressions.
The first is the weighted sum of the linear dependencies of pendulum deflection angles. 
It is awarded when the second pendulum is inclined from the vertical
by an angle of more than 10°. The values of the parameters of this sum were selected to
promote the swing of the second pendulum more than to align the first one. The
second formula works when the angle of the pendulum to the vertical is less than 10°. In
this phase, the agent must be concerned mainly with not losing his balance and moving the first pendulum closer to the target point. 
For this reason, this part of the function is a linear
dependence of only the angle of the first pendulum plus a penalty for loss of
balance.

### Environment parameters

This environment provides two parameters that can change the way it works.
- render_sim: (bool) if true, a graphic is generated
- n_steps: (int) number of time steps
- mode: (string) working mode, balance or swing"up
- m1: (float) the wieght of the first pendulum in [KG]
- m2: (float) the wieght of the second pendulum in [KG]
```
env = gym.make('double-pendulum-custom-v0', render_sim=False, n_steps=1000, mode = 'swing_up')
```
