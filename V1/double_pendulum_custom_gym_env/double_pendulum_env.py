from double_pendulum_custom_gym_env.event_handler import *
from double_pendulum_custom_gym_env.Pendulum import *

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
import os

class DoublePendulumEnv(gym.Env):
    """
    render_sim: (bool) if true, a graphic is generated
    n_steps: (int) number of time steps
    """

    def __init__(self, render_sim=False, n_steps=1000, m1 = 1, m2 = 1, L1 = 160, L2 = 160, mode = 'balance'):
        self.m1 = m1
        self.m2 = m2

        self.L1 = L1
        self.L2 = L2

        self.render_sim = render_sim

        # Working Mode
        self.mode = mode

        if self.render_sim is True:
            self.init_pygame()

        self.init_pymunk()

        #Parameters
        self.max_time_steps = n_steps
        if mode == 'balance':
            self.tourq_scale = 5000
        elif mode == 'swing_up':
            self.tourq_scale = 7000

        #Initial values
        self.tourq = 0
        self.done = False
        self.current_time_step = 0

        #Defining spaces for action and observation
        self.min_action = np.array([-1], dtype=np.float32)
        self.max_action = np.array([1], dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

        # alpha beta dalpha dbeta
        self.min_observation = np.array([-1, -1, -1, -1], dtype=np.float32)
        self.max_observation = np.array([1, 1, 1, 1], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.min_observation, high=self.max_observation, dtype=np.float32)

    def init_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Double Cartpole Environment")
        self.clock = pygame.time.Clock()

        script_dir = os.path.dirname(__file__)
        icon_path = os.path.join("img", "icon.png")
        icon_path = os.path.join(script_dir, icon_path)
        pygame.display.set_icon(pygame.image.load(icon_path))

    def init_pymunk(self):
        self.space = pymunk.Space()
        self.space.gravity = Vec2d(0, -1000)

        if self.render_sim is True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
            pymunk.pygame_util.positive_y_is_up = True

        initial_x = 400
        initial_y = 400
        self.target = 400


        # Center of movement
        self.COMove = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.COMove.position = (initial_x, initial_y)

        # Pendulum 1
        self.pole_1_mass = self.m1
        self.pole_1_length = self.L1
        pole_1_thickness = 15
        
        alpha = random.uniform(17*np.pi/36, 19*np.pi/36) + np.pi*(self.mode == 'swing_up')

        pole_1_x = initial_x + self.pole_1_length * np.cos(alpha)
        pole_1_y = initial_y + self.pole_1_length * np.sin(alpha)
        self.pole_1 = Pendulum(pole_1_x, pole_1_y, initial_x, initial_y, pole_1_thickness, self.pole_1_mass, (66, 135, 245), self.space)

        # Pendulum 2
        self.pole_2_mass = self.m2
        self.pole_2_length = self.L2
        pole_2_thickness = 15

        betha = random.uniform(17*np.pi/36, 19*np.pi/36) + np.pi*(self.mode == 'swing_up')

        pole_2_x = pole_1_x + self.pole_2_length * np.cos(betha)
        pole_2_y = pole_1_y + self.pole_2_length * np.sin(betha)
        self.pole_2 = Pendulum(pole_2_x, pole_2_y, pole_1_x, pole_1_y, pole_2_thickness, self.pole_2_mass, (119, 169, 248), self.space)

        self.pivot_1 = pymunk.PinJoint(self.COMove, self.pole_1.body, (0, 0), (-self.pole_1_length/2, 0))
        self.pivot_1.error_bias = 0
        self.pivot_1.collide_bodies = False
        self.space.add(self.pivot_1)

        self.pivot_2 = pymunk.PivotJoint(self.pole_1.body, self.pole_2.body, (self.pole_1_length/2, 0), (-self.pole_2_length/2, 0))
        self.pivot_2.error_bias = 0
        self.pivot_2.collide_bodies = False
        self.space.add(self.pivot_2)

    def step(self, action):
        self.tourq = action[0] * self.tourq_scale
        self.pole_1.body.apply_force_at_local_point((0, self.tourq), (0, 0))

        #Friction
        pymunk.Body.update_velocity(self.pole_1.body, Vec2d(0, 0), 0.999, 1/60.0)
        pymunk.Body.update_velocity(self.pole_2.body, Vec2d(0, 0), 0.99, 1/60.0)

        obs = self._get_observation()

        if np.abs(obs[1]) <= 1.0/18:
            stand = True
        else:
            stand = False

        self.space.step(1 / 60.0)
        self.current_time_step += 1

        obs = self._get_observation()

        if self.mode == 'balance':
            # Reward function
            reward = 2.21 - np.abs(obs[0])  - 0.1*np.abs(obs[2]) - 0.1*np.abs(obs[3]) - 0.01*np.abs(action[0])
            
            # Penalty for loss of balance
            if np.abs(obs[1]) > 20.0/180 or np.abs(obs[0]) > 20.0/180:
                self.done = True 
                reward = -50

        elif self.mode == 'swing_up':
            if np.abs(obs[1]) <= 1.0/18:
                reward = 10 + (-np.abs(obs[0])+1)*20
            else:
                reward = (18.0/17)*(-np.abs(obs[0]) + 1) + 0.5*(-np.abs(obs[1]) + 1)

            #Penalty for loss of balance
            if np.abs(obs[1]) > 1.0/18 and stand == True:
                reward = -50

        
        # Stops episode, when time is up
        if self.current_time_step == self.max_time_steps:
            self.done = True

        return obs, reward, self.done, {}

    def _get_observation(self):
        pole_1_angle = (((self.pole_1.body.angle+np.pi/2) % (2*np.pi)) + (2*np.pi)) % (2*np.pi)
        pole_1_angle = np.clip(-pole_1_angle/np.pi + 1, -1, 1, dtype=np.float32)

        pole_2_angle = (((self.pole_2.body.angle+np.pi/2) % (2*np.pi)) + (2*np.pi)) % (2*np.pi)
        pole_2_angle = np.clip(-pole_2_angle/np.pi + 1, -1, 1, dtype=np.float32)
        
        pole_1_angular_velocity = np.clip(self.pole_1.body.angular_velocity/15, -1, 1, dtype=np.float32)
        
        pole_2_angular_velocity = np.clip(self.pole_2.body.angular_velocity/15, -1, 1, dtype=np.float32)

        return np.array([pole_1_angle, pole_2_angle, pole_1_angular_velocity, pole_2_angular_velocity])

    def render(self, mode='human', close=False):
        scale = 1.0/200

        x, y = 400, 400
        pivot_point = self.pole_1.body.local_to_world([self.pole_1_length/2, 0])
        
        pygame_events()

        self.screen.fill((243, 243, 243))
        pygame.draw.line(self.screen, (149, 165, 166), (400, 0), (400, 800), 1)

        if self.mode == 'balance':
            x_prim = pivot_point[0] + (self.pole_2_length+25) * np.cos(7*np.pi/18)
            y_prim = pivot_point[1] + (self.pole_2_length+25) * np.sin(7*np.pi/18)
            pygame.draw.line(self.screen, (255, 26, 26), (pivot_point[0], 800-pivot_point[1]), (x_prim, 800-y_prim), 4)
            x_prim = pivot_point[0] + (self.pole_2_length+25) * np.cos(11*np.pi/18)
            y_prim = pivot_point[1] + (self.pole_2_length+25) * np.sin(11*np.pi/18)
            pygame.draw.line(self.screen, (255, 26, 26), (pivot_point[0], 800-pivot_point[1]), (x_prim, 800-y_prim), 4)

        self.space.debug_draw(self.draw_options)

        pygame.draw.circle(self.screen, (33, 93, 191), (x, y), 5)
        pygame.draw.circle(self.screen, (66, 135, 245), (pivot_point[0], 800-pivot_point[1]), 5)
        
        if self.tourq != 0:
            pivot_point = self.pole_1.body.local_to_world([0, 0])
            x = pivot_point[0]
            y = 800 - pivot_point[1]
            pygame.draw.line(self.screen, (255,0,0), (x, y), (x - scale*self.tourq*np.sin(self.pole_1.body.angle),y - scale*self.tourq*np.cos(self.pole_1.body.angle)), 4)

        pygame.display.flip()
        self.clock.tick(60)

    def reset(self):
        self.__init__(self.render_sim, self.max_time_steps, self.m1, self.m2, self.L1, self.L2, self.mode)
        return self._get_observation()

    def close(self):
        pygame.quit()
