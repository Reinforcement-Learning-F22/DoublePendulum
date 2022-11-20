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