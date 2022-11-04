# Import required libraries
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import cv2

from Pendulum import Pendulum

# set the parameter of our pendulum
m = 1 # mass of the pendulum bob
g = 9.81 # gravity acceleration
L = 0.2 # length of pendulum rod
I = 0.01 # inertia of actuator
b = 0.08 # friction in actuator 
dt = 0.01 # step value

# define the environment
env = Pendulum(m=m, L=L, I=I, b=b, dt=dt)
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

# Training the agent
env = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', env, verbose = 1)
model.learn(total_timesteps=20000)

# Evaluating the results of training 
evaluate_policy(model, env, n_eval_episodes=10, render=True)
cv2.waitKey(-1)
env.close()