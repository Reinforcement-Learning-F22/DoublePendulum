# Import required libraries
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import os

import numpy as np

from Pendulum import Pendulum

import matplotlib.pyplot as plt

# set the parameter of our pendulum
m = 1 # mass of the pendulum bob
g = 9.81 # gravity acceleration
L = 0.2 # length of pendulum rod
I = 0.01 # inertia of actuator
b = 0.08 # friction in actuator 
dt = 0.01 # step value

mode = 'swing_up'

# define the environment
env = Pendulum(m=m, L=L, I=I, b=b, dt=dt, mode=mode)
env.reset()

eval_rslt = []
eval_m = []
success = 0
max_n_episodes = 100
m_org = env.m

model_path = './models'
# Loading model after trainig
if mode == 'balance':
    PPO_path = os.path.join(model_path, 'Pendulum_balance_model')

    model = PPO.load(PPO_path, env=env)

    # Evaluating the model by changing the mass randomlly and plot the result
    for _ in range(0, max_n_episodes):
        env.m = m_org*(1 + 0.2*np.random.uniform(-1, +1))
        eval_m.append(env.m)
        eval_rslt.append(evaluate_policy(model, env, n_eval_episodes=1, render=False)[0])
        if eval_rslt[-1] > 195: 
            success += 1
        else:
            print(env.m)

    success = 100*success / max_n_episodes
    plt.plot(eval_rslt)
    plt.title(f'Evaluating of Pendulum in balance mode, success = {success}%')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()

elif mode == 'swing_up':
    PPO_path = os.path.join(model_path, 'Pendulum_swing_up_model')

    model = PPO.load(PPO_path, env=env)

    # Evaluating the model by changing the mass randomlly and plot the result
    for _ in range(0, max_n_episodes):
        env.m = m_org*(1 + 0.2*np.random.uniform(-1, +1))
        eval_m.append(env.m)
        eval_rslt.append(evaluate_policy(model, env, n_eval_episodes=1, render=False)[0])
        if eval_rslt[-1] > -1100: 
            success += 1
        else:
            print(env.m)

    success = 100*success / max_n_episodes
    plt.plot(eval_rslt)
    plt.title(f'Evaluating of Pendulum in swing_up mode, success = {success}%')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.show()
