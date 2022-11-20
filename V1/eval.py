from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gym
import time
import sys

import double_pendulum_custom_gym_env

import numpy as np

continuous_mode = True # if True, after completing one episode the next one will start automatically
random_action = False  # if True, the agent will take actions randomly

render_sim = True      # if True, a graphic is generated
evalute = False

if evalute:
    env = gym.make('double-pendulum-custom-v0', render_sim=False, n_steps=1000, L1 = 160, L2 = 160, mode='balance')

    model = PPO.load("./models/new_agent_final0.zip")
    model.set_env(env)

    random_seed = int(time.time())
    model.set_random_seed(random_seed)

    obs = env.reset()

    eval_rew, eval_len = evaluate_policy(model, env, n_eval_episodes=100, render=False, return_episode_rewards=True)
    print(f'Success with certainity= {np.average(eval_len)/10}%')

    eval_len = []
    for i in range(0, 10):
        # env = model.get_env()
        env.m2 = 1 + 0.5 * np.random.uniform(-1, +1)
        env.L2 = 160 + 0.5 * np.random.uniform(-160, +160)
        _, result = evaluate_policy(model, env, n_eval_episodes=1, render=False, return_episode_rewards=True)
        eval_len.append(result[0])

    print(f'Success with uncertainity = {np.average(eval_len)/10}%')

    env.close()

else:

    env = gym.make('double-pendulum-custom-v0', render_sim=render_sim, n_steps=1000, L1 = 160, L2 = 160, mode='swing_up')

    model = PPO.load("./models/new_agent_swing_upre_train.zip")
    model.set_env(env)

    random_seed = int(time.time())
    model.set_random_seed(random_seed)

    obs = env.reset()

    try:
        while True:
            if render_sim:
                env.render()

            if random_action:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs)

            obs, reward, done, info = env.step(action)
            if done is True:
                if continuous_mode is True:
                    state = env.reset()
                else:
                    break
    finally:
        env.close()
