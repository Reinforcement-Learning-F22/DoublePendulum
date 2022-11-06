# Import required libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import os

import cv2

from Pendulum import Pendulum

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

# Taking random actions and show the real time simulation
while True:
  # Take a random action
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  
  # Render the game
  env.render(mode = "human")
  
  if done == True:
    break
cv2.waitKey(2000)
env.close()

model = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', model, verbose = 1)

model_path = './models'
if mode == 'balance':
  model.learn(total_timesteps=20000)
  PPO_path = os.path.join(model_path, 'Pendulum_balance_model')
elif mode == 'swing_up':
  model.learn(total_timesteps=200000)
  PPO_path = os.path.join(model_path, 'Pendulum_swing_up_model')

# Saving model after trainig
model.save(PPO_path)

# Evaluating the results of training 
env.continues_run_mode = True
print(evaluate_policy(model, env, n_eval_episodes=1, render=True))
env.close()
