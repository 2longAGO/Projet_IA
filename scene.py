#import glob
import os
import sys
import subprocess
import collections
#import datetime
#import logging
import math
import random
import weakref
import numpy as np

v_kmh = lambda v: (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

class GenericSensor(object):
    SENSOR_HISTORY_BUFFER = 1
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
    
    def get_history(self):
        history = collections.defaultdict(int)
        for frame, data in self.history:
            history[frame] += data
        return history
    
    def append_history(self,data:tuple):
        if len(data) == 2:
            self.history.append(data)
            if len(self.history) > SENSOR_HISTORY_BUFFER:
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


class ObstacleSensor(GenericSensor):
    def __init__(self, parent_actor):
        super().__init__(parent_actor)
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ObstacleSensor._on_obstacle(weak_self, event))

    @staticmethod
    def _on_obstacle(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        #print('distance to: %r is %r' % (actor_type,event.distance))
        self.append_history((event.frame, event.distance))


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
        """
        lidar_range = 2.0*float(self.sensor.get_attribute('range'))

        points = np.frombuffer(lidar_measurement.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 6), 6))
        lidar_data = np.array(points[:, :2])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))

        # see nicely formatted points 
        for location in lidar_measurement:
            print(location)
        """
        #print(f"lidar: {lidar_measurement.raw_data}")
        self.append_history((lidar_measurement.frame,lidar_measurement.raw_data))


def clear_actors(to_destroy_list):
    for a in to_destroy_list :
        a.destroy()
    return []


def init_scene(args):
    actor_list = []
    if args.server_path is not None:
        # args for lower gpu load: -quality-level=Low 
        # -vulkan or -opengl
        subprocess.Popen(f'{args.server_path}')
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
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
                world = client.generate_opendrive_world(
                    data, carla.OpendriveGenerationParameters(
                        vertex_distance=vertex_distance,
                        max_road_length=max_road_length,
                        wall_height=wall_height,
                        additional_width=extra_width,
                        smooth_junctions=True,
                        enable_mesh_visibility=True))
            else:
                print('file not found.')

        sim_world = client.get_world()
        blueprint_library = sim_world.get_blueprint_library()
        # Spawn vehicle
        bp = sim_world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
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
        
        vehicle = sim_world.spawn_actor(bp,sim_world.get_map().get_spawn_points()[0])
        actor_list.append(vehicle)
        # Add sensors
        colSensor = CollisionSensor(vehicle)
        actor_list.append(colSensor)
        #obSensor = ObstacleSensor(vehicle)
        #actor_list.append(obSensor)
        sensor_points_per_second = '100000'
        lSensor = LiDARSensor(vehicle, carla.Transform(carla.Location(x=0, z=2.4)), {'channels' : '16', 'range' : '50', 'points_per_second': sensor_points_per_second, 'rotation_frequency': '20'})
        actor_list.append(lSensor)
        #rSensor = RadarSensor(vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(pitch=5, yaw=90)), {'points_per_second': sensor_points_per_second})
        #actor_list.append(rSensor)
        # Spawn obstacles
        for i in range(2):
            obstacle = random.choice(blueprint_library.filter("static.prop.*"))
            prop = sim_world.spawn_actor(obstacle,random.choice(sim_world.get_map().get_spawn_points()))
            actor_list.append(prop)
        
        return client,vehicle,sim_world,actor_list
    except:
        print("initialisation failed")
        clear_actors()
        client.load_world("town01")
        sys.exit()