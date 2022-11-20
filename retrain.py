# Import required libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

import os
import numpy as np
import cv2

from Pendulum2D import Pendulum2D
# set the parameter of our pendulum
L1 = 1
L2 = 0.5
m1 = 0.4
m2 = 0.15
J1 = 0.01
J2 = 0
b1 = 0.02
b2 = 0.002
g = 9.81
dt = 0.01
mode = 'swing_up'
# define the environment
env = Pendulum2D(m1=m1, m2=m2, L1=L1, L2=L2,
                 I1=J1, I2=J2, b1=b1, b2=b2,
                 dt=dt, mode=mode, max_itr=700)
env.reset()
'''
# Taking random actions and show the real time simulation
while True:
  # Take a random action
  action = env.action_space.sample()
  obs, reward, done, info = env.step(action)
  
  # Render the game
  env.render(mode = "human")
  
  if done == True:
    break

cv2.waitKey(500)'''
env.close()

model = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', model, verbose = 1, learning_rate=0.02 ,gamma=0.9999, device='cpu')


models_path = './models'
model_path=f"{models_path}/Pendulum2D_swing_up_model_40_ret_40_1.zip"

model= PPO.load(model_path, env=env)


if mode == 'balance':
  model.learn(total_timesteps=200000)
  PPO_path = os.path.join(models_path, 'Pendulum2D_balance_model')
elif mode == 'swing_up':
  model.learn(total_timesteps=5E4)
  PPO_path = os.path.join(models_path, 'Pendulum2D_swing_up_model_40_ret_40_3')
# Saving model after trainig
model.save(PPO_path)

# Evaluating the results of training 
# env.continues_run_mode = True
# print(evaluate_policy(model, env, n_eval_episodes=1, render=True))
env.close()

# Pendulum2D_swing_up_model_37_retrained_3 was reallly close, we can retrain it if theta2^4 does not do the trick