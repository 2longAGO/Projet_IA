"""
Requires CARLA to function 
When closing close the client first!!!
AFTER you can close the server if you please
"""


import glob
import os
import sys
import subprocess
"""
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
"""

import carla


import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import numpy as np

import pygame
from pygame.locals import*

actor_list = []  
SENSOR_HISTORY_BUFFER = 1000


class GenericSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
    
    def get_history(self):
        history = collections.defaultdict(int)
        for frame, data in self.history:
            history[frame] += intensity
        return history
    
    def append_history(self,data:tuple):
        if len(data) == 2:
            self.history.append(data)
            if len(self.history) > SENSOR_HISTORY_BUFFER:
                self.history.pop(0)
    
    def get_latest(self):
        return self.history[-1]

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
        #self.append_history()


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


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


def clear_actors():
    for a in actor_list :
        a.destroy()
    actor_list.clear()


def delete_multiple_lines(n=1):
    for _ in range(n):
        sys.stdout.write("\x1b[1A")  # cursor up one line
        sys.stdout.write("\x1b[2K")  # delete the last line


def init_scene(args):
    if args.server_path is not None:
        subprocess.Popen(args.server_path)
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
        
        return client,vehicle,sim_world
    except:
        print("initialisation failed")
        clear_actors()
        client.load_world("town01")
        sys.exit()


def loop(args):    
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    font = pygame.font.Font(pygame.font.get_default_font(), 16)
    client,vehicle,sim_world = init_scene(args)
    try:
        # Control vehicle 
        control = carla.VehicleControl(throttle=1.0,steer=0)
        # control.reverse boolean True = engaged False = disengaged
        # control.brake float 0 to 1
        vehicle.apply_control(control)
        orientation = 0.0
        while True :
            """
            if failure :
                clear_actors()
                client.reload_world()
                client,vehicle,sim_world = init_scene(args)
            """

            for event in pygame.event.get():
                if event.type == QUIT:       
                    pygame.quit()            
                    sys.exit()               
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()        
                        sys.exit()
            # get sensor data
            #vehicle.get_acceleration()
            #vehicle.get_velocity()
            # get speed in km/h
            v_kmh = lambda v: (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
            formatted_vec3d = lambda vec: 'Vector3d(x: {:2.3f} y: {:2.2f} z: {:2.3f})'.format(vec.x,vec.y,vec.z)
            # now print the text
            text_surface = font.render(f"speed: {'{:3.2f}'.format(v_kmh(vehicle.get_velocity()))} | {formatted_vec3d(vehicle.get_velocity())}", True, (255,255,255))
            screen.fill((0,0,0))
            screen.blit(text_surface, (20,50))
            pygame.display.update()
            sim_world.tick()
            # (TO-DO) Add Agent to make the desicions
    finally:
        print("End")
        clear_actors()
        client.load_world("town01")


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Collision Avoidance Training Client')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '-x', '--xodr-path',
        metavar='XODR_FILE_PATH',
        help='load a new map with a minimum physical road representation of the provided OpenDRIVE')
    argparser.add_argument(
        '-s', '--server-path',
        metavar='SERVER_FILE_PATH',
        help='path to server')
    args = argparser.parse_args()

    logging.info('listening to server %s:%s', args.host, args.port)

    #print(__doc__)

    try:

        loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()