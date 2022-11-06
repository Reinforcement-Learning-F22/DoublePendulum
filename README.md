# DoublePendulum
This is about make a RL agent to do the control of double pendulum and achieve the swing up and balancing problem.

## Prerequisites

* rotli>=1.0.9
* ConfigParser>=5.3.0
* cryptography>=38.0.3
* Cython>=0.29.32
* dl>=0.1.0
* docutils>=0.19
* gym>=0.21.0
* HTMLParser>=0.0.2
* importlib_metadata>=4.13.0
* ipaddr>=2.2.0
* keyring>=23.11.0
* lockfile>=0.12.2
* lxml>=4.9.1
* matplotlib>=3.6.1
* mypy_extensions>=0.4.3
* numpy>=1.23.4
* opencv_python>=4.6.0.66
* ordereddict>=1.1
* protobuf>=4.21.9
* pyOpenSSL>=22.1.0
* scipy>=1.7.1
* stable_baselines3>=1.6.2
* typing_extensions>=4.4.0
* wincertstore>=0.2.1
* xmlrpclib>=1.0.1
* zipp>=3.10.0

**All the libraries can be pip installed** using `python3 -m pip install -r requirements.txt`

## Getting Started 'Single Pendulum'

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
1. Navigate to repository folder
1. Install dependencies which are specified in requirements.txt. use `python3 -m pip install -r requirements.txt`
1. Run `project.py`.
1. Run `Uncertainity.py` if you want to test the model if there is uncertainity in mass of the pendulum.

# Single Pendulum
The project start with single pendulum, it is better to run it on the local machine, because cv2.imshow() won't work and will give an error.
You can set the parameter of your own system, it is very clear how to do that:
```
env = Pendulum(m=m, L=L, I=I, b=b, dt=dt, mode='balance')
```

## System parameters
```
m       # mass of the pendulum bob
L       # length of pendulum bob
I       # inertia of actuator
b       # friction in actuator 
g       # gravity acceleration
dt      # step size
theta   # initial angle
dtheta  # initial angular speed
mode    # working mode ['balance', 'swing_up']
max_itr # maximum iteration of episode balance = 200, swing_up = 500
```

## Pendulum mode:
If the mode is set to balance the pedulum will have the following consumption:
1. Start near the balance angle.
2. Get +1 reward if agent maintain he angle of pendulum between [-12, 12].
3. Terminate if get outside the angle range.
4. Maximum episode iteration will be 200 by default and the maximum return will be 200.

If the mode is set to swing_up the pedulum will have the following consumption:
1. Start near the down balance angle.
2. reward = -(2*theta^2 + 0.1*d_theta^2 + 0.01*tourq^2).
3. Terminate if the agent maintain the theta between [-12, 12] for time bigger than the half of maximum episode iteration.
4. Maximum episode iteration will be 500 by default.

## Test the system
For testing the system you can run the system by taking a random action from action space and apply it to the system, to do that you can use the following code:
```python
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
```
After running the previous you will got the following result:
![ezgif com-gif-maker](https://user-images.githubusercontent.com/90157234/200178563-60efb62e-be2b-4758-90f2-537f2d0f9f33.gif)

## Train the agent
We have now to train the agent, depend on the mode you can set the maximum number of iteration for training, swing_up mode is more general but also need much more time to be trained than the balance mode:
```python
model = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', model, verbose = 1)

model.learn(total_timesteps=20000)
```

## Test the agent
To test the agent we first activate the continues running mode:
```
env.continues_run_mode = True
```
In this mode the system will interact with the user, whom can use the keyboard to apply outside disturbance to the system, the user can use the arrows to increase or decrease the amount of external tourque she/he wants to apply, and the direction also, and also he can exit by pressinf any other key. 
```python
# Evaluating the results of training 
env.continues_run_mode = True
print(evaluate_policy(model, env, n_eval_episodes=1, render=True))
env.close()
```

```
i, up arrow      : increase the external tourque
d, down arrow    : decrease the external tourque
l, left arrow    : apply the external tourque to the left
r, right arrow   : apply the external tourque to the right
q, any other key : finish the testing
```

## Test with mass change
We cahnge the mass randomally (20%) then we evalute the model and we have success rate of 100%, and that is logical because the system is fully actuated and the only situation it could fail if the tourqe of the motor is not able to hold the mass.
<p align="center">
  <img src="https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/SP_Balance.png" />
</p>
