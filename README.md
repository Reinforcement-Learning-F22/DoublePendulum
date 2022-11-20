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
After running the previous you will got the following result if the mode is set to `balance`:

![ezgif com-gif-maker](https://user-images.githubusercontent.com/90157234/200178563-60efb62e-be2b-4758-90f2-537f2d0f9f33.gif)

And the following result if the mode is set to `swing_up`:

![swing_up_test](https://user-images.githubusercontent.com/90157234/200198293-bef5d29d-1d89-4676-8076-9f59ee8e6983.gif)

## Train the agent
We have now to train the agent, depend on the mode you can set the maximum number of iteration for training, swing_up mode is more general but also need much more time to be trained than the balance mode:
1. balance mode:
```python
model = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', model, verbose = 1)

model.learn(total_timesteps=20000)
```

2. swing_up mode:
```python
model = DummyVecEnv([lambda: env])
model = PPO('MlpPolicy', model, verbose = 1)

model.learn(total_timesteps=200000)
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
You can see the previous in the following window:
<p align="center">
  <img src="https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/SP_continues_mode.png" />
</p>

## Test with mass change
1. balance mode
We change the mass randomally (20%) then we evalute the model and we have success rate of 100%, and that is logical because the system is fully actuated and the only situation it could fail if the tourqe of the motor is not able to hold the mass.
<p align="center">
  <img src="https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/SP_Balance.png" />
</p>

2. swing_up mode
We change the mass randomally (20%) then we evalute the model and we have success rate of 100%, and that is logical because the system is fully actuated and the only situation it could fail if the tourqe of the motor is not able to hold the mass, but here it is very clear that the return is very connected to the value of the mass because the dynamics of the system will change, which mean the response of the system for any action will be diffferent, but the result is good enough and we won't made any improvment.
<p align="center">
  <img src="https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/SP_Swing_Up.png" />
</p>


## Dynamics of Double Pendulum (Pendubot)

<p align="center">
  <img src="https://diego.assencio.com/images/physics/double-pendulum.png" />
</p>

a double pendulum is a pendulum with another pendulum attached to its end, is a simple physical system that exhibits rich dynamic behavior with a strong sensitivity to initial conditions.

The motion of a double pendulum is governed by a set of coupled ordinary differential equations and is chaotic.

<table class="fraction">
  <tbody>
    <tr>
      <td rowspan="2"><i>θ</i><sub>1</sub>''&nbsp;=&nbsp;</td>
      <td>
        −<i>g</i> (2 <i>m</i><sub>1</sub> + <i>m</i><sub>2</sub>) sin <i>θ</i
        ><sub>1</sub> − <i>m</i><sub>2</sub> <i>g</i> sin(<i>θ</i><sub>1</sub> −
        2 <i>θ</i><sub>2</sub>) − 2 sin(<i>θ</i><sub>1</sub> − <i>θ</i
        ><sub>2</sub>) <i>m</i><sub>2</sub> (<i>θ</i><sub>2</sub>'<sup>2</sup>
        <i>L</i><sub>2</sub> + <i>θ</i><sub>1</sub>'<sup>2</sup> <i>L</i
        ><sub>1</sub> cos(<i>θ</i><sub>1</sub> − <i>θ</i><sub>2</sub>))
      </td>
    </tr>
    <tr>
      <td class="upper_line">
        <i>L</i><sub>1</sub> (2 <i>m</i><sub>1</sub> + <i>m</i><sub>2</sub> −
        <i>m</i><sub>2</sub> cos(2 <i>θ</i><sub>1</sub> − 2 <i>θ</i
        ><sub>2</sub>))
      </td>
    </tr>
  </tbody>
</table>

<table class="fraction">
  <tbody>
    <tr>
      <td rowspan="2"><i>θ</i><sub>2</sub>''&nbsp;=&nbsp;</td>
      <td>
        2 sin(<i>θ</i><sub>1</sub> − <i>θ</i><sub>2</sub>) (<i>θ</i
        ><sub>1</sub>'<sup>2</sup> <i>L</i><sub>1</sub> (<i>m</i><sub>1</sub> +
        <i>m</i><sub>2</sub>) + <i>g</i>(<i>m</i><sub>1</sub> + <i>m</i
        ><sub>2</sub>) cos <i>θ</i><sub>1</sub> + <i>θ</i><sub>2</sub>'<sup
          >2</sup
        >
        <i>L</i><sub>2</sub> <i>m</i><sub>2</sub> cos(<i>θ</i><sub>1</sub> −
        <i>θ</i><sub>2</sub>))
      </td>
    </tr>
    <tr>
      <td class="upper_line">
        <i>L</i><sub>2</sub> (2 <i>m</i><sub>1</sub> + <i>m</i><sub>2</sub> −
        <i>m</i><sub>2</sub> cos(2 <i>θ</i><sub>1</sub> − 2 <i>θ</i
        ><sub>2</sub>))
      </td>
    </tr>
  </tbody>
</table>

and after solving the differntial equations using scipy library;

```python
from scipy.integrate import odeint
sol = odeint(self.sys_ode, x0, [0, self.dt], args=(action, ))
self.theta1, self.theta2, self.dtheta1, self.dtheta2 = sol[-1, 0], sol[-1, 1], sol[-1, 2], sol[-1, 3]
```
and then simply plotting the results after calculating the positions of the masses we get:

![Double Pendulum Without Friction](https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/Simulation_no_f.gif)

but as you see we still have the problem of friction, without it the model is not realistic enough and moduling the previous equations into python, and for that we solve the problem by using Dynamics of Manipulators (we consider the Double Pendulum as a 2DOF manipulator).

The Equation of motion for most mechanical systems may be written in following form:

$Q=D(q)q¨+C(q,q˙)q˙+g(q)+Qd=D(q)q¨+c(q,q˙)+g(q)+Qd=D(q)q¨+h(q,q˙)+Qd$

>where:
>* $\mathbf{Q} \in \mathbb{R}^n$ - generalized forces corresponding to generilized coordinates
>* $\mathbf{Q}_d \in \mathbb{R}^n$ - generalized disippative forces (for instance friction)
>* $\mathbf{q} \in \mathbb{R}^{n}$ - vector of generilized coordinates
>* $\mathbf{D} \in \mathbb{R}^{n \times n}$ - positive definite symmetric inertia matrix 
>* $\mathbf{C} \in \mathbb{R}^{n \times n}$ - describe 'coefficients' of centrifugal and Coriolis forces
>* $\mathbf{g} \in \mathbb{R}^{n}$ - describes effect of gravity and other position depending forces
>* $\mathbf{h} \in \mathbb{R}^n$ - combined effect of $\mathbf{g}$ and $\mathbf{C}$

In order to find the EoM we will use the Lagrange-Euler equations:

$d/dt(∂L/∂q˙i)−∂L/∂qi=Qi−∂R/∂q˙i,i=1,2,…,n$

>where:
>* $\mathcal{L}(\mathbf{q},\dot{\mathbf{q}}) \triangleq E_K - E_\Pi \in \mathbb{R}$ Lagrangian of the system  
>* $\mathcal{R} \in \mathbb{R}$ Rayleigh function  (describes energy dissipation)


and here we add two dissipative elements in this system, namely "dampers" with coefficients  $b1,b2$  (viscous friction), their dissipation function is given as:

$\mathcal{R} = \frac{1}{2}\ ∑  b_j \dot{\alpha}^2_j$

and after applying Lagrange formalism to obtain equations of motion;

$I_1\ddot{\alpha}_1 + l_1^2 (m_1 + m_2) \ddot{\alpha}_1 + l_1 l_2 m_2 \cos(\alpha_1 - \alpha_2)\ddot{\alpha}_2 +$ 
$l_1 l_2 m_2 \sin(\alpha_1 - \alpha_2)\dot{\alpha}^2_2$
$+l_1 m_1 g \cos \alpha_1 + l_1 m_2 g \cos \alpha_2 + b_1 \dot{\alpha}_1 =u_1$
$l_1 l_2 m_2 \cos(\alpha_1 - \alpha_2)\ddot{\alpha}_1 + I_2 \ddot{\alpha} + l_2^2 m_2 \ddot{\alpha}_2 - l_2 m_2 l_1 \sin(\alpha_1 - \alpha_2)\dot{\alpha}^2_1 + l_2 m_2 g \cos \alpha_2+ b_2 \dot{\alpha}_2 = u_2$

Now we can find the  $D,C,g$ . all details are in the code DynamicsDP.py.

and so we get:

![Double Pendulum With Friction](https://github.com/Reinforcement-Learning-F22/DoublePendulum/blob/main/img/Simulation_f.gif)

## Training the Double Pendulum - Balancing

for balancing the task is simple enough to be solved by a the idea of a simple reward function; since the DP is starting from around the vertical position, we just give negatice reward for speeds and the theta1 and theta2 to be far away from the vertical position as follows;

## Training the Double Pendulum - Swing_up
we've tried over 50 different reward functions (the models and the some of the reward functions are uploaded) to make it swing up AND balance! and we couldn't succeed until we used an if statement in the reward function, and for continuous input RL problems, continuous Reward functions might not work!
for example one of the reward functions (that will be explained in details);


$reward= - ( a ({\pi}- \theta_2)^4+ b (\theta_1)^2 + c (\dot{\theta_1})^2 (1.1- cos(\theta_1)) + d \dot{\theta_2}^2 (1.1- cos(\theta_1)) + e 0.01 Tourq^2)$


it was trained for 10,000,000 timesteps for 24 hours and it could not swing up and balance, it could only swing up.
then it occured to us since it seems that it needs 2 agents, one for swing up and the other for balancing vertically , we decided to use if statement in the reward function
