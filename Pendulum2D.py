from gym import Env, spaces
#gym.logger.set_level(40)
#action_space = spaces.Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), dtype=np.float32)
import cv2

import numpy as np
from scipy.integrate import odeint

from DynamicsDP import *

# Define a custome environment based on the "gym" environment
# That will make us able to use the same training and evaluting functions that used with standard gym libaries
class Pendulum2D(Env):
    # Initializing function where to define the vallue of system parameters
    def __init__(self, m1, m2, L1, L2, I1, I2, b1, b2, 
                 theta1 = 0, dtheta1 = 0, theta2 = 0, dtheta2 = 0, 
                 dt = 0.01, g = 9.81, mode='balance', max_itr = -1):
        super(Pendulum2D, self).__init__()
        
        # System parameters
        self.m1 = m1 # mass of the pendulum first bob
        self.L1 = L1 # length of pendulum first rod
        self.I1 = I1 # inertia of the first actuator
        self.b1 = b1 # friction in the first actuator 

        self.m2 = m2 # mass of the pendulum second bob
        self.L2 = L2 # length of pendulum second rod
        self.I2 = I2 # inertia of the second actuator
        self.b2 = b2 # friction in the second actuator 

        self.g = g # gravity acceleration
        self.dt = dt # step size

        # Set timing parameters including total time and max iteration of the episode
        self.t = 0
        self.max_itr = max_itr

        # Angle and angular speed of the pedulum
        self.theta1 = theta1
        self.dtheta1 = dtheta1

        self.theta2 = theta2
        self.dtheta2 = dtheta2
        
        self._Max_dtheta1 = 4 * np.pi 
        self._Max_dtheta2 = 9 * np.pi
        self._Max_tourq = 4

        # The control of the system is tourq
        self.tourq = np.array([0])
        # Mode of working where we have two modes 
        # 1. balance : the starting angle will be near balance and the agent must keep the pendulum balanced.
        # 2. swing_up : the starting angle will be near pi/2, and the agent must bring the pendulum to balance position.
        if mode in ['balance', 'swing_up']:
          self.mode = mode
        else:
          self.mode = 'balance'
          print(f'{mode} is not correct, mode set to balance' )

        # Observation of the system are [angle of the first mass, angular speed of the first mass, angle of the second mass, angular speed of the second mass]
        self.observation = [self.theta1, self.dtheta1, self.theta2, self.dtheta2]

        # Define the observation space
        self.observation_shape = (4, )
        self.observation_space = spaces.Box(low  = np.array([-np.pi, -4*np.pi, -np.pi, -9*np.pi]), 
                                            high = np.array([np.pi, +4*np.pi, np.pi, +9*np.pi]),
                                            dtype = np.float32) 
        
        # Define an action space ranging from -2 to 2, which is the amount of tourque that must be applied
        self.action_space = spaces.Box(low  =np.array([-1.0]),
                                       high =np.array([+1.0]),
                                       shape = (1, ),
                                       dtype = np.float32)

        # Using CV2 is more effecient than ploting
        self.canvas = np.ones((600, 800, 3))
        # Calculate the ration for drawing
        self._ratio1 = (self.L1)/(self.L1 + self.L2)
        self._ratio2 = (self.L2)/(self.L1 + self.L2)

        # Set variable for continue runing
        self.continues_run_mode = False
        self.external_tourq = 1
        self.apply_external_tourq = 0
    
    # This function is for drawing the pendulum, the output will be (600, 800, 3) uint8 array
    def system_plot(self):
        temp = np.ones((600, 800, 3), dtype=np.uint8) * 255
      
        # Draw the center
        temp = cv2.circle(temp, (400,300), 3, (0,0,255), -1)

        x1 = int(400 + 200 * self._ratio1 * np.sin(self.theta1))
        y1 = int(300 + 200 * self._ratio1 * np.cos(self.theta1))
      
        # draw the beam1
        temp = cv2.line(temp, (400, 300), (x1, y1), (0, 0, 0), 1)

        # Draw the mass1
        temp = cv2.circle(temp, (x1, y1), 10, (255,0,0), -1)

        x2 = x1 + int(200 * self._ratio2 * np.sin(self.theta1 + self.theta2))
        y2 = y1 + int(200 * self._ratio2 * np.cos(self.theta1 + self.theta2))
      
        # draw the beam1
        temp = cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # Draw the mass1
        temp = cv2.circle(temp, (x2, y2), 10, (255,0,0), -1)
      
        # Display the values of angle, angular speed and tourq on the simulation window
        temp = cv2.rectangle(temp, (580, 470), (790, 590), (0, 0,0), 2)
        temp = cv2.putText(temp, "theta1: " + str(round(self.theta1, 3)), (585, 490), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
        temp = cv2.putText(temp, "dtheta1: " + str(round(self.dtheta1, 3)), (585, 510), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
        temp = cv2.putText(temp, "theta2: " + str(round(self.theta2, 3)), (585, 530), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
        temp = cv2.putText(temp, "dtheta2: " + str(round(self.dtheta2, 3)), (585, 550), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)
        temp = cv2.putText(temp, "tourq: " + str(self.tourq), (585, 570), cv2.FONT_HERSHEY_PLAIN, 1.5, 255)

        # Add instruction on the screan when run on continues running mode
        if self.continues_run_mode:  
            temp = cv2.rectangle(temp, (10, 490), (575, 590), (0, 0,0), 2)
            temp = cv2.putText(temp, "Press i to increase tourq, Press d to decrease tourq", (15, 520), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            temp = cv2.putText(temp, "Press l to apply positive tourq, Press r to apply negative tourq", (15, 540), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            temp = cv2.putText(temp, "External tourq: " + str(self.external_tourq), (15, 560), cv2.FONT_HERSHEY_PLAIN, 1, 255)
            temp = cv2.putText(temp, "Press any other key to exit", (15, 580), cv2.FONT_HERSHEY_PLAIN, 1, 255)

        self.canvas = temp

    # This is the function for reseting the system
    def reset(self):
      if self.mode == 'balance':
        # set the initial angles to random value
        self.theta1  = np.pi + np.random.uniform(-0.05, 0.05)
        self.theta2  = np.random.uniform(-0.05, 0.05)
        # self.theta1  = np.random.uniform(-0.05, 0.05)
        # self.theta2  = np.random.uniform(-0.05, 0.05)

        # set the angular velocities to random value
        self.dtheta1 = np.random.uniform(-0.01, 0.01)
        self.dtheta2 = np.random.uniform(-0.01, 0.01)

        # Setting the default episode lenght in balance mode
        if self.max_itr == -1: self.max_itr = 200

      elif self.mode == 'swing_up':
        # set the initial angles to random value near the down balance angle
        self.theta1  = np.random.uniform(-0.05, 0.05)
        self.theta2  = np.random.uniform(-0.05, 0.05)

        # set the angular velocities to random value
        self.dtheta1 = np.random.uniform(-1.0, 1.0)
        self.dtheta2 = np.random.uniform(-1.0, 1.0)

        # Setting the default episode lenght in swing up mode
        if self.max_itr == -1: self.max_itr = 500

        # Define the life time for termination criterial
        self.alive = 0 

      while self.theta1 > np.pi:
          self.theta1 = self.theta1 - 2*np.pi 
        
      while self.theta1 < -np.pi:
        self.theta1 = self.theta1 + 2*np.pi

      while self.theta2 > np.pi:
          self.theta2 = self.theta2 - 2*np.pi 
        
      while self.theta2 < -np.pi:
        self.theta2 = self.theta2 + 2*np.pi

      # Set the initial action
      self.tourq = np.array([0])

      # Set the observations
      self.observation = [self.theta1, self.dtheta1, self.theta2, self.dtheta2]

      # set time to 0
      self.t = 0

      # Reset the reward
      self.ep_return = 0
      
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
        key = cv2.waitKey(1) & 0xFF

        # if we are in run mode we check the kes pressed by the user
        if self.continues_run_mode:
          if key == ord('i') or key == 0:
            # key i pressed
            self.external_tourq += 0.1
          elif key == ord('d') or key == 1:
            # key d pressed
            self.external_tourq -= 0.1
          elif key == ord('r') or key == 3:
            # key r pressed
            self.apply_external_tourq = +1
          elif key == ord('l') or key == 2:
            # key l pressed
            self.apply_external_tourq = -1
          elif key != 255:
            # Exit at any key
            self.apply_external_tourq = -2
          return self.canvas
          
      elif mode == "rgb_array":
        return self.canvas
    
    # Close all cv2 windows
    def close(self):
      cv2.destroyAllWindows()
      cv2.waitKey(1)

    # Function that define the solution of the differential equation of our system
    def sys_ode(self, x, t, action):
        q, dq = (x[:2]), x[2:4]

        params = (self.L1, self.L2), (self.m1, self.m2), (self.I1, self.I2), (self.b1, self.b2), self.g

        D_c = D(q, params)
        h_c = h(q, dq, params)
        Q_d_c = Q_d(q, dq, params)

        ddq = np.dot(np.linalg.inv(D_c), action - Q_d_c -  h_c )

        dx1 = dq
        dx2 = ddq
        dx = dx1, dx2

        return np.concatenate(dx)

    # We will define the step function which return the new state, reward and termenal flag for a given action
    def step(self, action):
        # Flag that marks the termination of an episode
        done = False
        info = {}
        self.t += self.dt
        # Assert that it is a valid action 
        assert self.action_space.contains(action), "Invalid Action"

        self.tourq = action[0] * self._Max_tourq

        # apply the action to the system
        x0 = [self.theta1, self.theta2, self.dtheta1, self.dtheta2]

        # if we are in run mode we check the pressed key in the render function
        if self.continues_run_mode:
            # if the pressed key is not related to tourque apply or change we end the simulation
            if self.apply_external_tourq == -2:
                return self.observation, 0, True, {}
                
            action = self.tourq, -1*self.external_tourq*self.apply_external_tourq
            self.apply_external_tourq = 0
            sol = odeint(self.sys_ode, x0, [0, self.dt], args=(action, ))
            self.theta1, self.theta2, self.dtheta1, self.dtheta2 = sol[-1, 0], sol[-1, 1], sol[-1, 2], sol[-1, 3]

            while self.theta1 > np.pi:
                self.theta1 = self.theta1 - 2*np.pi 
        
            while self.theta1 < -np.pi:
                self.theta1 = self.theta1 + 2*np.pi

            while self.theta2 > np.pi:
                self.theta2 = self.theta2 - 2*np.pi 
        
            while self.theta2 < -np.pi:
                self.theta2 = self.theta2 + 2*np.pi

            self.observation = [self.theta1, self.dtheta1, self.theta2, self.dtheta2]

            if self.mode == 'balance':
              reward = +1
              if abs(sin(self.theta1 + self.theta2)) > 0.3:
                reward = -1
            if self.mode == 'swing_up':
              reward = +1
              if abs((self.theta1 + self.theta2)) < 0.2:
                reward = -1
                # self.reset()
            print(reward)
            # Increment the episodic return
            self.ep_return += 1
      
        # elif self.mode == 'swing_up':
        #   reward = -(2*self.theta**2 + 0.1*self.dtheta**2 + 0.01*action**2)

        # Draw the new state
            self.system_plot()
            return self.observation, reward, False, {}

        else:
            action = self.tourq, 0

            sol = odeint(self.sys_ode, x0, [0, self.dt], args=(action, ))
      
            self.theta1, self.theta2, self.dtheta1, self.dtheta2 = sol[-1, 0], sol[-1, 1], sol[-1, 2], sol[-1, 3]

            while self.theta1 > np.pi:
                self.theta1 = self.theta1 - 2*np.pi 
        
            while self.theta1 < -np.pi:
                self.theta1 = self.theta1 + 2*np.pi

            while self.theta2 > np.pi:
                self.theta2 = self.theta2 - 2*np.pi 
        
            while self.theta2 < -np.pi:
                self.theta2 = self.theta2 + 2*np.pi

            self.observation = [self.theta1, self.dtheta1, self.theta2, self.dtheta2]

            # the reward depend on the mode of the pendulum
            # 1. balance : reward is 1 as well as the pendulum is between [-12, +12] degree.
            # 2. swing_up : reward is a reflection of how much the system is doing well 
            # so the [angle, angular speed, tourge] must be as small as possible, also we ass some weight
            # which make the angle have more influent than the other parameters 
            
            if self.mode == 'balance':
                # Termination if the pendulum is outside the angle range
                # reward = -(10*(self.L1 + self.L2 + self.L1 * np.cos(self.theta1) + self.L2 * np.cos(self.theta1 + self.theta2))**2 + 0.1*(self.dtheta1**2 + self.dtheta2**2))
                reward = (+4 + (abs(self.theta1)/np.pi)**2 - ((self.dtheta1/(4*np.pi))**2 + (self.dtheta2/(9*np.pi))**2) - (self.tourq/4)**2)
                # x = self.L1*np.sin(self.theta1) + self.L2*np.sin(self.theta1 + self.theta2)
                # y = self.L1*np.cos(self.theta1) + self.L2*np.cos(self.theta1 + self.theta2)
                # reward = -(x**2 + (y + self.L1 + self.L2)**2 + 0.1*(self.dtheta1**2 + self.dtheta2**2) + 0.01*action[0]**2)

                if abs(sin(self.theta1 + self.theta2)) > 0.3:
                    info["TimeLimit.truncated"] = False
                    if self.ep_return + 1 > 0.95*self.max_itr:
                      pass
                    reward = -1
                    done = True
      
            elif self.mode == 'swing_up':
              
              #if np.abs(self.theta1)<2.70:
              
                #reward = -( 10 * np.abs(np.abs(self.theta1)-np.pi)**2  + 10 * np.abs(self.theta2) + 0.1* np.abs(self.tourq) )
              
              # second best #reward =  -( 64 *((np.abs(self.theta1)-np.pi)**2)**2 + 4 * (self.theta2)**2 + 4 *np.abs(self.dtheta1)* (self.dtheta1)**2  + (self.dtheta2)**2 + 0.1* (self.tourq)**2 )/100
              #reward =  -( 64 *((np.abs(self.theta1)-np.pi)**2)**2 + 8 * (self.theta2)**2 + 8 *np.abs(self.dtheta1)* (self.dtheta1)**2)  + (self.dtheta2)**2 + 0.1* (self.tourq)**2 )/100 #Pendulum2D_swing_up_model_37 reward # Best yet    

              #reward = -(((self.theta2)**2**2))  #Pendulum2D_swing_up_model_37 reward # Best yet    
              
              #BEST#
              # reward =  -( 20 *((np.abs(self.theta1)-np.pi)**2)**2 + 10 *((((self.theta2)**2)**2)**2)*((1-np.cos(self.theta1)**2)) + 4 *(np.abs(self.dtheta1)* (self.dtheta1)**2)*((1-np.cos(self.theta1)**2))**2  + ((self.dtheta2)**2)*((1-np.cos(self.theta1)**2))**2 + 0.1* (self.tourq)**2 )/100 # devieation possible solution for the Best yet # 1-cos(theta1)    
             
              reward =  -( 20 *((np.abs(self.theta1)-np.pi)**2)**2 + 10 *((((self.theta2)**2)**2)**2)*((1-np.cos(self.theta1)**2)) + 6 *((self.dtheta1**2) * (self.dtheta1)**2)*((1-np.cos(self.theta1)**2))**2  + ((self.dtheta2)**2)*((1-np.cos(self.theta1)**2))**2 + 0.1* (self.tourq)**2 )/100
              
              ''' #Could be tried, a bit promising..
              if ( np.abs(self.theta1)> 2.9 and np.abs(self.theta2) < 0.12 ):
                reward+= 200 - np.power(self.dtheta1,4)
              if ( np.abs(self.theta1)> 3.1 and np.abs(self.theta2) < 0.04 ):
                reward+= 200 - (self.dtheta1)**2

              if (np.abs(self.dtheta1) > 3 and np.abs(self.theta1)> 2.3):
                reward -= 300000
                done=True'''
              
              
              
              #reward =  -( 32 *((np.abs(self.theta1)-np.pi)**2)**2 + 8 * np.abs(self.theta2)*(self.theta2)**2 + 4 *np.abs(self.dtheta1)* (self.dtheta1)**2  + (self.dtheta2)**2 + 0.1* (self.tourq)**2 )/100 #Pendulum2D_swing_up_model_37_retrained_3  really close
              #reward =  -( 64 *((np.abs(self.theta1)-np.pi)**2)**2 + 16 * (self.theta2)**2 + 6 *np.abs(self.dtheta1)* (self.dtheta1)**2  + (self.dtheta2)**2 + 0.1* (self.tourq)**2 )/100  # 38
              
              #reward = 2* (((self.theta1)**2)**2)+ 2* ((((np.pi - np.abs(self.theta2))*self.theta1)**2)**2)- (self.tourq)**2-(self.dtheta1)**2**2
              
              #pre-last#reward =  np.power((self.theta1/np.pi),8) + np.power((((np.pi - np.abs(self.theta2)))/(np.pi)),8) 
              
              #last#reward= np.power(self.theta1,4) + (np.pi-self.theta2)*self.theta1 - (np.pi-(self.theta1+self.theta2))
              
              #reward =  np.power((self.theta1/np.pi),8) + np.power((((np.pi - np.abs(self.theta2)))/(np.pi)),8) - np.power( (self.dtheta1/(4*np.pi)),8) - (np.abs(self.theta1)-np.pi)/np.pi
              
              #else:
              
                #reward = (+4 + (abs(self.theta1)/np.pi)**2 - ((self.dtheta1/(4*np.pi))**2 + (self.dtheta2/(9*np.pi))**2) - (self.tourq/4)**2)
              
              #reward= (self.theta2)**2
              
                # Adding 1 to allive if the pendulum is in angle range
              
              if (np.abs(self.theta1)>2.92 and np.abs(self.theta2) < 0.12):
                  self.alive += 1
              else:
                  self.alive += 0

                # Increment the episodic return
                  self.ep_return = self.alive
        
                # If alive time is larger than 200 iteration then it is done
              if self.alive >= self.max_itr / 2:
                    done = True

            # if self.ep_return < -300:
            #     print(self.ep_return)
            #     # reward = -3
            #     # done = True

            # The lenght of the episode is more than the calculated lenght
            if self.t >= self.max_itr * self.dt:
                info["TimeLimit.truncated"] = True
                # print('Timeout')
                # reward = -1E10 ## was commented
                done = True
            
            # Draw the new state
            self.system_plot()
            self.ep_return += reward
            return self.observation, reward, done, info