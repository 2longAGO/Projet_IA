# fmt: off
"""
Make your own custom environment
================================

This documentation overviews creating new environments and relevant
useful wrappers, utilities and tests included in Gymnasium designed for
the creation of new environments. You can clone gym-examples to play
with the code that is presented here. We recommend that you use a virtual environment:

.. code:: console

   git clone https://github.com/Farama-Foundation/gym-examples
   cd gym-examples
   python -m venv .env
   source .env/bin/activate
   pip install -e .

Subclassing gymnasium.Env
-------------------------

Before learning how to create your own environment you should check out
`the documentation of Gymnasium’s API </api/env>`__.

We will be concerned with a subset of gym-examples that looks like this:

.. code:: sh

   gym-examples/
     README.md
     setup.py
     gym_examples/
       __init__.py
       envs/
         __init__.py
         grid_world.py
       wrappers/
         __init__.py
         relative_position.py
         reacher_weighted_reward.py
         discrete_action.py
         clip_reward.py

To illustrate the process of subclassing ``gymnasium.Env``, we will
implement a very simplistic game, called ``GridWorldEnv``. We will write
the code for our custom environment in
``gym-examples/gym_examples/envs/grid_world.py``. The environment
consists of a 2-dimensional square grid of fixed size (specified via the
``size`` parameter during construction). The agent can move vertically
or horizontally between grid cells in each timestep. The goal of the
agent is to navigate to a target on the grid that has been placed
randomly at the beginning of the episode.

-  Observations provide the location of the target and agent.
-  There are 4 actions in our environment, corresponding to the
   movements “right”, “up”, “left”, and “down”.
-  A done signal is issued as soon as the agent has navigated to the
   grid cell where the target is located.
-  Rewards are binary and sparse, meaning that the immediate reward is
   always zero, unless the agent has reached the target, then it is 1.

An episode in this environment (with ``size=5``) might look like this:

where the blue dot is the agent and the red square represents the
target.

Let us look at the source code of ``GridWorldEnv`` piece by piece:
"""

# %%
# Declaration and Initialization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our custom environment will inherit from the abstract class
# ``gymnasium.Env``. You shouldn’t forget to add the ``metadata``
# attribute to your class. There, you should specify the render-modes that
# are supported by your environment (e.g. ``"human"``, ``"rgb_array"``,
# ``"ansi"``) and the framerate at which your environment should be
# rendered. Every environment should support ``None`` as render-mode; you
# don’t need to add it in the metadata. In ``GridWorldEnv``, we will
# support the modes “rgb_array” and “human” and render at 4 FPS.
#
# The ``__init__`` method of our environment will accept the integer
# ``size``, that determines the size of the square grid. We will set up
# some variables for rendering and define ``self.observation_space`` and
# ``self.action_space``. In our case, observations should provide
# information about the location of the agent and target on the
# 2-dimensional grid. We will choose to represent observations in the form
# of dictionaries with keys ``"agent"`` and ``"target"``. An observation
# may look like ``{"agent": array([1, 0]), "target": array([0, 3])}``.
# Since we have 4 actions in our environment (“right”, “up”, “left”,
# “down”), we will use ``Discrete(4)`` as an action space. Here is the
# declaration of ``GridWorldEnv`` and the implementation of ``__init__``:
import math
import glob
import os
import sys
import subprocess
import numpy as np

import pygame
from pygame.locals import*

import gymnasium as gym
from gymnasium import spaces

import time
import math
import random
import re
import weakref
import carla
import collections

# Scene code 
v_kmh = lambda v: (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class GenericSensor(object):

    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.SENSOR_HISTORY_BUFFER = 1
    
    def get_history(self):
        history = collections.defaultdict(int)
        for frame, data in self.history:
            history[frame] += data
        return history
    
    def append_history(self,data:tuple):
        if len(data) == 2:
            self.history.append(data)
            if len(self.history) > self.SENSOR_HISTORY_BUFFER:
                self.history.pop(0)
    
    def get_latest(self):
        if len(self.history) > 0 :
            return self.history[-1]
        else:
            return None

    def destroy(self):
        self.sensor.destroy()


class CollisionSensor(GenericSensor):
    def __init__(self, parent_actor):
        super().__init__(parent_actor)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        #print('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.append_history((event.frame, intensity))


class LiDARSensor(GenericSensor):
    def __init__(self, parent_actor,transform,sensor_options):
        super().__init__(parent_actor)
        world = self._parent.get_world()
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
        lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        
        for key in sensor_options:
            lidar_bp.set_attribute(key, sensor_options[key])
        
        self.sensor = world.spawn_actor(lidar_bp, transform, attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LiDARSensor.lidar_data(weak_self, event))

    def lidar_data(weak_self, lidar_measurement):
        self = weak_self()
        if not self:
            return
        #print(f"lidar: {lidar_measurement.raw_data}")
        self.append_history((lidar_measurement.frame,lidar_measurement.raw_data))
    
    def get_lidar_image(self,window_size=512,range=50,clamp=False):
        disp_size = [window_size,window_size]
        lidar_range = 2.0*float(range)
        data = self.get_latest()[1]
        points = np.frombuffer(data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3) if not clamp else (disp_size[0], disp_size[1],1)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255) if not clamp else 1
        return lidar_img

class RadarSensor(GenericSensor):
    def __init__(self, parent_actor,transform,sensor_options):
        super().__init__(parent_actor)
        world = self._parent.get_world()
        radar_bp = world.get_blueprint_library().find('sensor.other.radar')
        for key in sensor_options:
            radar_bp.set_attribute(key, sensor_options[key])
        self.sensor = world.spawn_actor(radar_bp, transform, attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: RadarSensor.radar_data(weak_self, event))

    def radar_data(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        #print('Collision with %r' % radar_data)
        self.append_history((radar_data.frame,radar_data))
    # (TO-DO) make better get_radar_image function
    def get_radar_image(self,window_size=512,rotation=math.pi):
        radar_data = self.get_latest()[1]
        velocity_range = 7.5
        #points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        #points = np.reshape(points, (len(radar_data), 4))
        #print(points)
        """
        carla.Transform(
            carla.Location(),
            carla.Rotation(
                pitch=current_rot.pitch + alt,
                yaw=current_rot.yaw + azi,
                roll=current_rot.roll)).transform(fw_vec)
        """
        disp = pygame.Surface((window_size,window_size-20))
        center = (window_size/2,(window_size-20)/2)
        lines = []
        for data in radar_data:
            azi = math.degrees(data.azimuth)
            alt = math.degrees(data.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=data.depth - 0.25)
            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))
            norm_velocity = data.velocity / velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            vec = pygame.math.Vector3(radar_data.transform.location.x + fw_vec.x,radar_data.transform.location.y + fw_vec.y,radar_data.transform.location.z + fw_vec.z).normalize()
            #vec_data = pygame.math.Vector3.from_spherical((data.depth,data.altitude,data.azimuth+rotation))
            lines.append(center)
            #disp.set_at((int(vec_data.x+center[0]*3),int(vec_data.z+center[1]*3)), (255,255,255))
            #lines.append(((vec_data.x+center[0])*1.8,(vec_data.z+center[1])*1.8))
            disp.set_at((int((vec.x+center[0])*1.5),int((vec.z+center[1])*1.5)), (r,g,b))
            #lines.append(((fw_vec.x+center[0])*1.5,(fw_vec.z+center[1])*1.5))
        #if len(lines) > 1 :
        #    pygame.draw.aalines(disp,(255,255,255),False,lines)

        current_rot = radar_data.transform.rotation
        #print(len(radar_data))
        return disp


def clear_actors(to_destroy_list):
    for a in to_destroy_list :
        a.destroy()
    return []


class CarlaAvoidanceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    actor_list = []
    world = None
    process = None
    def __init__(self, args, render_mode=None, nbObstacles=2):
        self.window_size = 512  # The size of the PyGame window
        self.nbObstacles = nbObstacles

        if args.server_path is not None:
            # args for lower gpu load: -quality-level=Low 
            # -vulkan or -opengl
            self.process = subprocess.Popen(f'{args.server_path}')
        
        try:
            self.client = carla.Client(args.host, args.port)
            self.client.set_timeout(20.0)

            # Lire le fichier de terrain si ajouté
            if args.xodr_path is not None:
                if os.path.exists(args.xodr_path):
                    with open(args.xodr_path, encoding='utf-8') as od_file:
                        try:
                            data = od_file.read()
                        except OSError:
                            print('file could not be readed.')
                            sys.exit()
                    print('load opendrive map %r.' % os.path.basename(args.xodr_path))
                    vertex_distance = 2.0  # in meters
                    max_road_length = 500.0 # in meters
                    wall_height = 1.0      # in meters
                    extra_width = 0.6      # in meters
                    world = self.client.generate_opendrive_world(
                        data, carla.OpendriveGenerationParameters(
                            vertex_distance=vertex_distance,
                            max_road_length=max_road_length,
                            wall_height=wall_height,
                            additional_width=extra_width,
                            smooth_junctions=True,
                            enable_mesh_visibility=True))
                elif args.town is not None and any(re.search(args.town, town) for town in self.client.get_available_maps()) :
                    self.client.load_world(args.town)
                else:
                    print('file not found.')

            self.world = self.client.get_world()
            self.original_settings = self.world.get_settings()
            self.sync = args.sync
            if args.sync:
                settings = world.get_settings()
                traffic_manager.set_synchronous_mode(True)
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            blueprint_library = self.world.get_blueprint_library()
            # Spawn vehicle
            bp = blueprint_library.filter("vehicle.tesla.model3")[0]
            bp.set_attribute('role_name', "hero")
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            if bp.has_attribute('driver_id'):
                driver_id = random.choice(bp.get_attribute('driver_id').recommended_values)
                bp.set_attribute('driver_id', driver_id)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'true')
            # set the max speed
            player_max_speed = 1.589
            player_max_speed_fast = 3.713
            if bp.has_attribute('speed'):
                player_max_speed = float(bp.get_attribute('speed').recommended_values[1])
                player_max_speed_fast = float(bp.get_attribute('speed').recommended_values[2])
            else:
                print("No recommended values for 'speed' attribute")
            self._agent_location, self._target_location = [self.world.get_map().get_waypoint(spawn.location).transform for spawn in np.random.choice(self.world.get_map().get_spawn_points(), 2, replace=False)]
            self.vehicle = self.world.spawn_actor(bp,self.world.get_map().get_spawn_points()[0])
            self.actor_list.append(self.vehicle)
            # Add sensors
            self.colSensor = CollisionSensor(self.vehicle)
            self.actor_list.append(self.colSensor)
            #obSensor = ObstacleSensor(self.vehicle)
            #self.actor_list.append(obSensor)
            sensor_points_per_second = '100000'
            self.channels = 64
            self.range = 10
            self.lSensor = LiDARSensor(self.vehicle, carla.Transform(carla.Location(x=0, z=2.4)), {'channels' : f'{self.channels}', 'range' : f'{self.range}', 'points_per_second': sensor_points_per_second, 'rotation_frequency': '20'})
            self.actor_list.append(self.lSensor)
            self.rSensor = RadarSensor(self.vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(pitch=0, yaw=0)), {'points_per_second': sensor_points_per_second,'vertical_fov': str(20), 'horizontal_fov': str(70),'range' : str(self.range)})
            self.actor_list.append(self.rSensor)
            # Spawn obstacles
            for i in range(self.nbObstacles):
                obstacle = random.choice(blueprint_library.filter("static.prop.*"))
                prop = self.world.spawn_actor(obstacle,random.choice(self.world.get_map().get_spawn_points()))
                self.actor_list.append(prop)
        except:
            print("initialisation failed")
            clear_actors(self.actor_list)
            if self.world is not None:
                self.client.load_world("town01")
            if self.process is not None:
                self.process.terminate()
            sys.exit()

        # init pygame 
        self.window = pygame.init()
        self.screen = pygame.display.set_mode((self.window_size,self.window_size))
        self.font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.img_size = 64
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        # discrete states wall left right front back front-left front-right back-left back-right if less than half range away they are true
        self.observation_space = spaces.Dict(
            {
                "obstacles": spaces.Box(0, 1, shape=(1,self.img_size**2), dtype=np.float32), # lidar observation space
                #"obstacles": spaces.Box(0, self.range, shape=(self.channels,), dtype=np.float32), # swap high for range of sensor and add second value to shape for number of rays
                "speed": spaces.Box(0, 500, dtype=np.float32), # high = get max speed from vehicle initialisation
                "distTarget": spaces.Box(0, sys.maxsize, dtype=np.float32),
            }
        )

        #print(self.observation_space.is_np_flattenable)
        self.action_space = gym.spaces.Discrete(4)
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.ACT_AMT = 0.25
        # self.action_space = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32) # steer, throttle(+), brake(-) Non discrete values

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

# %%
# Constructing Observations From Environment States
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Since we will need to compute observations both in ``reset`` and
# ``step``, it is often convenient to have a (private) method ``_get_obs``
# that translates the environment’s state into an observation. However,
# this is not mandatory and you may as well compute observations in
# ``reset`` and ``step`` separately:

    def _get_obs(self):
        ldata = self.lSensor.get_lidar_image(window_size=self.img_size, range=self.range, clamp=True),
        #ldata = [self.range]*self.channels if not isinstance(ldata, tuple) or not isinstance(ldata, list) or ldata is None else [carla.Location.distance(carla.Location(data[0],data[1],data[2])) for data in ldata[1]]
        #ldata = [[self.range,self.range,self.range]]*self.channels if not isinstance(ldata, tuple) or not isinstance(ldata, list) or ldata is None else [[data[0],data[1],data[2]] for data in ldata[1]]
        #rdata = [data.depth for data in self.rSensor.get_latest()[1]][:self.channels]
        dataArr = np.array(ldata,dtype=np.float32)
        dataArr = np.ndarray.flatten(dataArr)
        return {"obstacles": dataArr, "speed": v_kmh(self.vehicle.get_velocity()), "distTarget": self._agent_location.location.distance(self._target_location.location)}

# %%
# We can also implement a similar method for the auxiliary information
# that is returned by ``step`` and ``reset``. In our case, we would like
# to provide the manhattan distance between the agent and the target:

    def _get_info(self):
        return { "agent": self._agent_location.location, "target": self._target_location.location }

# %%
# Oftentimes, info will also contain some data that is only available
# inside the ``step`` method (e.g. individual reward terms). In that case,
# we would have to update the dictionary that is returned by ``_get_info``
# in ``step``.

# %%
# Reset
# ~~~~~
#
# The ``reset`` method will be called to initiate a new episode. You may
# assume that the ``step`` method will not be called before ``reset`` has
# been called. Moreover, ``reset`` should be called whenever a done signal
# has been issued. Users may pass the ``seed`` keyword to ``reset`` to
# initialize any random number generator that is used by the environment
# to a deterministic state. It is recommended to use the random number
# generator ``self.np_random`` that is provided by the environment’s base
# class, ``gymnasium.Env``. If you only use this RNG, you do not need to
# worry much about seeding, *but you need to remember to call
# ``super().reset(seed=seed)``* to make sure that ``gymnasium.Env``
# correctly seeds the RNG. Once this is done, we can randomly set the
# state of our environment. In our case, we randomly choose the agent’s
# location and the random sample target positions, until it does not
# coincide with the agent’s position.
#
# The ``reset`` method should return a tuple of the initial observation
# and some auxiliary information. We can use the methods ``_get_obs`` and
# ``_get_info`` that we implemented earlier for that:

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.init_scene()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def init_scene(self):
         # Do a soft reset (teleport vehicle)
        control = carla.VehicleControl(steer=float(0.0),throttle=float(0.0),brake=float(0.0))
        self.vehicle.apply_control(control)
        #self.vehicle.tick()
        self.colSensor.history.clear()
        self.lSensor.history.clear()
        self.rSensor.history.clear()
        
        # Generate waypoints along the lap
        self._agent_location, self._target_location = [self.world.get_map().get_waypoint(spawn.location).transform for spawn in np.random.choice(self.world.get_map().get_spawn_points(), 2, replace=False)]
        self.vehicle.set_transform(self._agent_location)
        #(TO-DO) Add check for car acceleration on reset to make sure it is set in place and the physics didn't go haywire
        self.vehicle.set_simulate_physics(False) # Reset the car's physics
        self.vehicle.set_simulate_physics(True)

        # Give 2 seconds to reset
        if self.sync:
            ticks = 0
            while ticks < self.fps * 2:
                self.world.tick()
                try:
                    self.world.wait_for_tick(seconds=1.0)
                    ticks += 1
                except:
                    pass
        else:
            time.sleep(2.0)
    
# %%
# Step
# ~~~~
#
# The ``step`` method usually contains most of the logic of your
# environment. It accepts an ``action``, computes the state of the
# environment after applying that action and returns the 4-tuple
# ``(observation, reward, done, info)``. Once the new state of the
# environment has been computed, we can check whether it is a terminal
# state and we set ``done`` accordingly. Since we are using sparse binary
# rewards in ``GridWorldEnv``, computing ``reward`` is trivial once we
# know ``done``. To gather ``observation`` and ``info``, we can again make
# use of ``_get_obs`` and ``_get_info``:

    def step(self, action):
        #(TO-DO) Complete the step function
        self.tick_world()
        # Control vehicle 
        # Continous states
        # control = carla.VehicleControl(steer=action[0],throttle=max(action[1],0),brake=-min(action[1],0))

        # Discrete states
        if action == 0:
            self.throttle = min(self.throttle+self.ACT_AMT,1)
            self.brake = 0
        elif action == 1: # left?
            self.steer = self.steer + self.ACT_AMT
        elif action == 2: # right?
            self.steer = self.steer - self.ACT_AMT
        elif action == 3:
            self.brake = min(self.throttle+self.ACT_AMT,1)
            self.throttle = 0
        self.steer = min(max(self.steer,-1),1)
        control = carla.VehicleControl(steer=self.steer,throttle=self.throttle,brake=self.brake)


        # control.reverse boolean True = engaged False = disengaged
        # control.brake float 0 to 1
        self.vehicle.apply_control(control)
        self._agent_location = self.vehicle.get_transform()
        reward = 0
        terminated = False
        # An episode is done if the agent has reached the target
        # terminated = np.array_equal(self._agent_location.location, self._target_location.location)
        reward = 50 if np.array_equal(self._agent_location.location, self._target_location.location) else 0  # Binary sparse rewards
        reward += 1 if v_kmh(self.vehicle.get_velocity()) > 50 else 0 #max(min(50-v_kmh(self.vehicle.get_velocity()),1),0)

        if self.colSensor.get_latest() is not None :
            reward = -200 
            terminated = True
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def tick_world(self):
        # Carla Tick
        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def rad_callback(self, radar_data):
        velocity_range = 7.5 # m/s
        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.world.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))
# %%
# Rendering
# ~~~~~~~~~
#
# Here, we are using PyGame for rendering. A similar approach to rendering
# is used in many environments that are included with Gymnasium and you
# can use it as a skeleton for your own environments:

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        self.tick_world()


        for event in pygame.event.get():
            if event.type == QUIT:       
                self.close()            
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()        
                    sys.exit()
                elif event.key == K_1:
                    settings = self.world.get_settings()
                    settings.no_rendering_mode = True
                    self.world.apply_settings(settings)
                elif event.key == K_2:
                    settings = self.world.get_settings()
                    settings.no_rendering_mode = False
                    self.world.apply_settings(settings)
        """
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl(steer=float(0.0),throttle=float(0.0),brake=float(0.0),reverse=False)
        if keys[K_w]:
            control.throttle=float(0.5)
        if keys[K_s]:
            control.reverse = True
        if keys[K_d]:
            control.steer=float(0.5)
        elif keys[K_a]:
            control.steer=float(-0.5)
        self.vehicle.apply_control(control)
        """
        # get speed in km/h
        formatted_vec3d = lambda vec: 'Vector3d(x: {:2.3f} y: {:2.2f} z: {:2.3f})'.format(vec.x,vec.y,vec.z)
        # now print the text
        text_surface = self.font.render(f"speed: {'{:3.2f}'.format(v_kmh(self.vehicle.get_velocity()))} | {formatted_vec3d(self.vehicle.get_velocity())}", True, (255,255,255))
        self.screen.fill((0,0,0))
        self.rad_callback(self.rSensor.get_latest()[1])
        self.screen.blit(self.rSensor.get_radar_image(rotation=math.radians(self.vehicle.get_transform().rotation.yaw)),(0,20))
        self.screen.blit(pygame.surfarray.make_surface(self.lSensor.get_lidar_image(window_size=self.window_size)),(0,0))
        self.screen.blit(text_surface, (0,0))
        pygame.display.update()
        return np.array(pygame.surfarray.array3d(self.screen), dtype=np.uint8).transpose([1, 0, 2])

# %%
# Close
# ~~~~~
#
# The ``close`` method should close any open resources that were used by
# the environment. In many cases, you don’t actually have to bother to
# implement this method. However, in our example ``render_mode`` may be
# ``"human"`` and we might need to close the window that has been opened:

    def close(self):
        clear_actors(self.actor_list)
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        if self.world is not None:
            self.client.load_world("town01")
        if self.process is not None:
            self.process.terminate()
        sys.exit()


# %%
# In other environments ``close`` might also close files that were opened
# or release other resources. You shouldn’t interact with the environment
# after having called ``close``.

# %%
# Registering Envs
# ----------------
#
# In order for the custom environments to be detected by Gymnasium, they
# must be registered as follows. We will choose to put this code in
# ``gym-examples/gym_examples/__init__.py``.
#
# .. code:: python
#
#   from gymnasium.envs.registration import register
#
#   register(
#        id="gym_examples/GridWorld-v0",
#        entry_point="gym_examples.envs:GridWorldEnv",
#        max_episode_steps=300,
#   )

# %%
# The environment ID consists of three components, two of which are
# optional: an optional namespace (here: ``gym_examples``), a mandatory
# name (here: ``GridWorld``) and an optional but recommended version
# (here: v0). It might have also been registered as ``GridWorld-v0`` (the
# recommended approach), ``GridWorld`` or ``gym_examples/GridWorld``, and
# the appropriate ID should then be used during environment creation.
#
# The keyword argument ``max_episode_steps=300`` will ensure that
# GridWorld environments that are instantiated via ``gymnasium.make`` will
# be wrapped in a ``TimeLimit`` wrapper (see `the wrapper
# documentation </api/wrappers>`__ for more information). A done signal
# will then be produced if the agent has reached the target *or* 300 steps
# have been executed in the current episode. To distinguish truncation and
# termination, you can check ``info["TimeLimit.truncated"]``.
#
# Apart from ``id`` and ``entrypoint``, you may pass the following
# additional keyword arguments to ``register``:
#
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | Name                 | Type      | Default   | Description                                                                                                   |
# +======================+===========+===========+===============================================================================================================+
# | ``reward_threshold`` | ``float`` | ``None``  | The reward threshold before the task is  considered solved                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``nondeterministic`` | ``bool``  | ``False`` | Whether this environment is non-deterministic even after seeding                                              |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``max_episode_steps``| ``int``   | ``None``  | The maximum number of steps that an episode can consist of. If not ``None``, a ``TimeLimit`` wrapper is added |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``order_enforce``    | ``bool``  | ``True``  | Whether to wrap the environment in an  ``OrderEnforcing`` wrapper                                             |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``autoreset``        | ``bool``  | ``False`` | Whether to wrap the environment in an ``AutoResetWrapper``                                                    |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
# | ``kwargs``           | ``dict``  | ``{}``    | The default kwargs to pass to the environment class                                                           |
# +----------------------+-----------+-----------+---------------------------------------------------------------------------------------------------------------+
#
# Most of these keywords (except for ``max_episode_steps``,
# ``order_enforce`` and ``kwargs``) do not alter the behavior of
# environment instances but merely provide some extra information about
# your environment. After registration, our custom ``GridWorldEnv``
# environment can be created with
# ``env = gymnasium.make('gym_examples/GridWorld-v0')``.
#
# ``gym-examples/gym_examples/envs/__init__.py`` should have:
#
# .. code:: python
#
#    from gym_examples.envs.grid_world import GridWorldEnv
#
# If your environment is not registered, you may optionally pass a module
# to import, that would register your environment before creating it like
# this - ``env = gymnasium.make('module:Env-v0')``, where ``module``
# contains the registration code. For the GridWorld env, the registration
# code is run by importing ``gym_examples`` so if it were not possible to
# import gym_examples explicitly, you could register while making by
# ``env = gymnasium.make('gym_examples:gym_examples/GridWorld-v0)``. This
# is especially useful when you’re allowed to pass only the environment ID
# into a third-party codebase (eg. learning library). This lets you
# register your environment without needing to edit the library’s source
# code.

# %%
# Creating a Package
# ------------------
#
# The last step is to structure our code as a Python package. This
# involves configuring ``gym-examples/setup.py``. A minimal example of how
# to do so is as follows:
#
# .. code:: python
#
#    from setuptools import setup
#
#    setup(
#        name="gym_examples",
#        version="0.0.1",
#        install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
#    )
#
# Creating Environment Instances
# ------------------------------
#
# After you have installed your package locally with
# ``pip install -e gym-examples``, you can create an instance of the
# environment via:
#
# .. code:: python
#
#    import gym_examples
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#
# You can also pass keyword arguments of your environment’s constructor to
# ``gymnasium.make`` to customize the environment. In our case, we could
# do:
#
# .. code:: python
#
#    env = gymnasium.make('gym_examples/GridWorld-v0', size=10)
#
# Sometimes, you may find it more convenient to skip registration and call
# the environment’s constructor yourself. Some may find this approach more
# pythonic and environments that are instantiated like this are also
# perfectly fine (but remember to add wrappers as well!).
#
# Using Wrappers
# --------------
#
# Oftentimes, we want to use different variants of a custom environment,
# or we want to modify the behavior of an environment that is provided by
# Gymnasium or some other party. Wrappers allow us to do this without
# changing the environment implementation or adding any boilerplate code.
# Check out the `wrapper documentation </api/wrappers/>`__ for details on
# how to use wrappers and instructions for implementing your own. In our
# example, observations cannot be used directly in learning code because
# they are dictionaries. However, we don’t actually need to touch our
# environment implementation to fix this! We can simply add a wrapper on
# top of environment instances to flatten observations into a single
# array:
#
# .. code:: python
#
#    import gym_examples
#    from gymnasium.wrappers import FlattenObservation
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = FlattenObservation(env)
#    print(wrapped_env.reset())     # E.g.  [3 0 3 3], {}
#
# Wrappers have the big advantage that they make environments highly
# modular. For instance, instead of flattening the observations from
# GridWorld, you might only want to look at the relative position of the
# target and the agent. In the section on
# `ObservationWrappers </api/wrappers/#observationwrapper>`__ we have
# implemented a wrapper that does this job. This wrapper is also available
# in gym-examples:
#
# .. code:: python
#
#    import gym_examples
#    from gym_examples.wrappers import RelativePosition
#
#    env = gymnasium.make('gym_examples/GridWorld-v0')
#    wrapped_env = RelativePosition(env)
#    print(wrapped_env.reset())     # E.g.  [-3  3], {}
