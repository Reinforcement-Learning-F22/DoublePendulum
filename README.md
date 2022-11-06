# DoublePendulum
This is about make a RL agent to do the control of double pendulum and achieve the swing up and balancing problem.

The project start with single pendulum, it is better to run it on the local machine, because cv2.imshow() won't work and will give an error.
You can set the parameter of your own system, it is very clear how to do that:
env = Pendulum(m=m, L=L, I=I, b=b, dt=dt, mode='balance')

# System parameters
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

# Pendulum mode:
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
