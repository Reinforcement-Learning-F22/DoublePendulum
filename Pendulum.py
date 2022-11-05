from gym import Env, spaces

import cv2

import numpy as np
from scipy.integrate import odeint

# Define a custome environment based on the "gym" environment
# That will make us able to use the same training and evaluting functions that used with standard gym libaries
class Pendulum(Env):
    # Initializing function where to define the vallue of system parameters
    def __init__(self,m , L, I, b, dt = 0.01, theta = 0, dtheta = 0, t = 0, g = 9.81, mode='balance', max_itr = -1):
        super(Pendulum, self).__init__()
        
        # System parameters
        self.m = m # mass of the pendulum bob
        self.L = L # length of pendulum rod
        self.I = I # inertia of actuator
        self.b = b # friction in actuator 
        self.g = g # gravity acceleration
        self.dt = dt # step size

        # # we can calculate the position of the mass 
        # self.x = L * np.sin(theta)
        # self.y = L * np.cos(theta)

        # Set timing parameters including total time and max iteration of the episode
        self.t = t
        self.max_itr = max_itr

        # Angle and angular speed of the pedulum
        self.theta = theta
        self.dtheta = dtheta
        
        # The control of the system is tourq
        self.tourq = np.array([0])

        # Mode of working where we have two modes 
        # 1. balance : the starting angle will be near balance and the agent must keep the pendulum balanced.
        # 2. swing_up : any random angle between [-pi/2, pi/2], and the agent must bring the pendulum to balance position.
        self.mode = mode

        # Observation of the system are [angle of the mass, angular speed of the mass]
        self.observation = [self.theta, self.dtheta]

        # Define the observation space
        self.observation_shape = (2, )
        self.observation_space = spaces.Box(low  = np.array([-np.pi/2, -8]), 
                                            high = np.array([+np.pi/2, +8]),
                                            dtype = np.float32)
    
        # Using CV2 is more effecient than ploting
        self.canvas = np.ones((600, 800, 3)) 


        # Define an action space ranging from -2 to 2, which is the amount of tourque that must be applied
        self.action_space = spaces.Box(low  =np.array([-2.0]),
                                       high =np.array([+2.0]),
                                       shape = (1, ),
                                       dtype = np.float32)
    
    # This function is for drawing the pendulum, the output will be (600, 800, 3) uint8 array
    def system_plot(self):
      temp = np.ones((600, 800, 3), dtype=np.uint8) * 255
      
      # Draw the center
      temp = cv2.circle(temp, (400,300), 3, (0,0,255), -1)

      x = int(400 - 250 * np.sin(self.theta))
      y = int(300 - 250 * np.cos(self.theta))
      
      # draw the beam
      temp = cv2.line(temp, (400, 300), (x, y), (0, 0, 0), 1)

      # Draw the mass
      temp = cv2.circle(temp, (x, y), 10, (255,0,0), -1)

      # Display the values of angle, angular speed and tourq on the simulation window
      temp = cv2.rectangle(temp, (590, 490), (790, 590), (0, 0,0), 2)
      temp = cv2.putText(temp, "theta: " + str(round(self.theta, 3)), (595, 530), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
      temp = cv2.putText(temp, "dtheta: " + str(round(self.dtheta, 3)), (595, 550), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
      temp = cv2.putText(temp, "tourq: " + str(round(self.tourq[0], 3)), (595, 570), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
      self.canvas = temp

    # This is the function for reseting the system
    def reset(self):
      if self.mode == 'balance':
        # set the initial_angle to random value
        self.theta  = np.random.uniform(-0.05, 0.05)

        # set the angular velocity to random value
        self.dtheta = np.random.uniform(-1.0, 1.0)

        # Setting the default episode lenght in balance mode
        if self.max_itr: self.max_itr = 200

      elif self.mode == 'swing_up':
        # set the initial_angle to random value near the down balance angle
        self.theta  = np.pi + np.random.uniform(-0.05, 0.05)

        # set the angular velocity to random value
        self.dtheta = np.random.uniform(-1.0, 1.0)

        # Setting the default episode lenght in swing up mode
        if self.max_itr: self.max_itr = 500

        # Define the life time for termination criterial
        self.alive = 0 

      # Set the initial action
      self.tourq = np.array([0])

      # Set the observations
      self.observation = [self.theta, self.dtheta]

      # set time to 0
      self.t = 0

      # Reset the reward
      self.ep_return  = 0
      
      # reset the plotting
      self.system_plot()

      # return the observation
      return self.observation

    # This function either simulate the system at real time or return the array of the user
    def render(self, mode = "human"):
      assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
    
      if mode == "human":
        cv2.imshow('Pendulum', self.canvas)
        cv2.setWindowProperty('Pendulum', cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(10)
    
      elif mode == "rgb_array":
        return self.canvas
    
    # Close all cv2 windows
    def close(self):
      cv2.destroyAllWindows()

    # Function that define the solution of the differential equation of our system
    def sys_ode(self, x, t, u):
        theta = x[0]
        dtheta = x[1]
        ddtheta = (u - self.m * self.g * self.L * np.sin(theta) - self.b * dtheta)/(self.I + self.m * self.L**2) # acceleration in joint 
        dx = dtheta, ddtheta # concatenate dtheta, and ddtheta to the state derevitive dx
        return dx

    # We will define the step function which return the new state, reward and termenal flag for a given action
    def step(self, action):
      # Flag that marks the termination of an episode
      done = False
      self.tourq = action
      # Assert that it is a valid action 
      assert self.action_space.contains(action), "Invalid Action"

      # apply the action to the system
      x0 = [self.theta, self.dtheta]
      sol = odeint(self.sys_ode, x0, [0, self.dt], args=(action, ))
      self.theta, self.dtheta = sol[-1,0], sol[-1,1]
      self.t += self.dt

      self.observation = [self.theta, self.dtheta]

      # the reward depend on the mode of the pendulum
      # 1. balance : reward is 1 as well as the pendulum is between [-12, +12] degree.
      # 2. swing_up : reward is a reflection of how much the system is doing well 
      # so the [angle, angular speed, tourge] must be as small as possible, also we ass some weight
      # which make the angle have more influent than the other parameters 
      if self.mode == 'balance':
        reward = 1

        # Termination if the pendulum is outside the angle range
        if np.abs(self.theta) > np.pi * 12.0 / 180:
          done = True

        # Increment the episodic return
        self.ep_return += 1
      
      elif self.mode == 'swing_up':
        reward = -(2*self.theta**2 + 0.1*self.dtheta**2 + 0.001*action**2)
        
        # Adding 1 to allive if the pendulum is in angle range
        if np.abs(self.theta) < np.pi * 12.0 / 180:
          self.alive += 1
        else:
          self.alive += 0

        # Increment the episodic return
        self.ep_return = self.alive
        
        # If alive time is larger than 200 iteration then it is done
        if self.alive >= self.max_itr / 2:
          done = True

      # The lenght of the episode is more than the calculated lenght
      if self.t >= self.max_itr * self.dt:
        done = True

      # Draw the new state
      self.system_plot()

      return self.observation, reward, done, {}
