#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys





try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass


import carla

from carla import ColorConverter as cc
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from agents.navigation.controller_edited import VehiclePIDController

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

import networkx as nx


import tensorflow as tf       # Deep Learning library
import numpy as np            # Handle matrices


import random                 # Handling random number generation
import time                   # Handling time calculation
from skimage import transform # Helps in preprocessing the frames

from collections import deque # Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs


import evdev
from evdev import ecodes, InputDevice


import Reward
import Q_network
import pickle

import Simulator










most_recent_image=None
episodes=5
process_semantic_image = False
initial_point=carla.Location(x=-85.0, y= 13.0, z=8.0)
goal_point=carla.Location(x=12.0, y= 190.0, z=1.0)








if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


'''


self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]


'''

class World(object):
    def __init__(self, carla_world, hud, actor_filter, route, route_adversary):
        #settings = carla_world.get_settings()
        #settings.no_rendering_mode = True
        #carla_world.apply_settings(settings)
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.Route=route
        self.Route_adversary=route_adversary
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = actor_filter
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.next_weather()

    def restart(self):
        # Keep same camera config if the camera manager exists.
        #cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_index=5 #0 to change to rgb cam
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 1 #0 to change camera view
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.player is None:
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            #spawn_point = carla.Transform


            all_actors= self.world.get_actors()
            all_vehicles = all_actors.filter('vehicle.*') 
            
            while len(all_vehicles)<1:
                all_actors= self.world.get_actors()
                all_vehicles = all_actors.filter('vehicle.*') 
            
            for x in all_vehicles:
                  if x==None:
                     protagonist_point=self.Route[min(len(self.Route)-1, 20)]
                     protagonist_point.rotation.yaw=-self.Route[min(len(self.Route)-1, 20)].rotation.yaw
                     print("no other cars!")
                  else :
                      protagonist_point=x.get_transform()
                      #print("found ",x)
            
            spawn_adversary=self.adversary_initialization(protagonist_point, spawn_points)
            #print(spawn_adversary)

            #spawn_point.location.x = 12.0
            #spawn_point.location.y = 190.0
            #spawn_point.location.z = 2.0
            #spawn_point.rotation.yaw = -90.0
            #spawn_point.rotation.roll = 0.0
            #spawn_point.rotation.pitch = 0.0
            #self.player = self.world.spawn_actor(blueprint, spawn_adversary) 
            #spawn_point.location.x = -85.0
            #spawn_point.location.y = 13.0
            #spawn_point.location.z = 10.0
            #spawn_point.rotation.yaw = -180.0
            #spawn_point.rotation.roll = 0.0
            #spawn_point.rotation.pitch = 0.0
            

            while self.player==None:
                self.player = self.world.try_spawn_actor(blueprint, spawn_point) 
                spawn_point = random.choice(spawn_points)
            #spawn_point = carla.Transform
            #self.player.set_transform(self.get_transform_adversary(protagonist_point.location,-protagonist_point.rotation.yaw,d= 30))
            location=spawn_adversary.location
            angle=spawn_adversary.rotation.yaw
            spawn_adversary=carla.Transform(location, carla.Rotation(yaw=180+angle))
            
            self.player.set_transform(spawn_adversary)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)


    '''
    def get_transform_adversary(self,vehicle_location, angle, d=30):
        a = math.radians(angle)
        location = carla.Location(d * math.cos(a), d * math.sin(a), 3.0) + vehicle_location
        return carla.Transform(location, carla.Rotation(yaw=180+angle))
    '''
    
    def adversary_initialization(self, spawn_point, spawn_points):
         #dis=100
         finish=False
         ix=0
         spawn_location=spawn_point.location
         dis_prev=1000000
         while not finish:
             dis_current=np.linalg.norm([spawn_location.x - self.Route[ix].location.x , spawn_location.y-self.Route[ix].location.y])
            # print(dis_current,ix)
             if dis_current>dis_prev:
                  finish=True
             else:
                dis_prev=dis_current
             ix=ix+1
             if ix>len(self.Route):
                 finish=True
         n=60  # Initialize adversary 60 waypoints ahead of the protagonist
         adversary_point=self.Route[min(len(self.Route)-1, ix+n)]    # Not the center lane route but we don't care; Add the center route function if needed
         adversary_point.rotation.yaw=self.Route[min(len(self.Route)-1, ix+n)].rotation.yaw
         #distance=1000
         #prev_distance= 10000
         #for point in spawn_points:
         #    distance=np.linalg.norm([adversary_point.location.x - point.location.x , adversary_point.location.y-point.location.y])
         #    if distance<prev_distance:
         #        closest_spawn_point=point
         #        prev_distance=distance
         #print(distance)
         #closest_spawn_point.rotation.yaw=self.Route[min(len(self.Route)-1, ix+100)].rotation.yaw
         return adversary_point   #closest_spawn_point

    #def adversary_initialization2(self, spawn_point, spawn_points):
    #    pass

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1 
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)
       
        '''
        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))
       '''
    def parse_events(self, world, clock, action=None):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                #self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                #self._parse_vehicle_wheel()
                # Major Control Edits
                self._control=Reward.choose_control(action, self._control)
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)
            return False
            

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        if pygame.font.get_fonts() == [None]:
            fonts='ubuntumono'
        else:
            fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            #'Map:     % 20s' % world.world.map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        self.actor_collision_list=[]
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        self.actor_collision_list.append(event.other_actor)
        if len(self.history) > 4000:
            self.history.pop(0)
            self.actor_collision_list.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        global most_recent_image
        self = weak_self()
        if not self:
            return 
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            if process_semantic_image: 
               self.surface = pygame.surfarray.make_surface(Q_network.simplify_semantic_image(array).swapaxes(0, 1))
               #plt.imshow(Q_network.simplify_semantic_image(array))
               #plt.show()
            else:
               self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            array=Q_network.simplify_semantic_image(array)
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)
        most_recent_image = array

# ==============================================================================
# -- test_environment() ---------------------------------------------------------------
# ==============================================================================


def test_environment(args):
    for i in range(episodes):
        game_loop(args, memory=False, explore_exploit_action=False, decay_step=0, episode=0, tau=0, DQNetwork=None, TargetNetwork=None, writer=None, write_op=None, training=False, sess=None)
        time.sleep(2)
    return





def choose_pid_waypoint_adversary(route,world):
    #if index>len(route)-1:
    #        index=len(route)-1
    #pid_waypoint = route[index]
    #player_loc = player.get_location()
    #world=player.get_world()
    _,ix =Reward.closestRoutePoint_adversary(world,start_index=0)
    index=ix+5
    if index>len(route)-1:
       index=len(route)-1
          #  break
    return index 



# ==============================================================================
# -- Sum tree for Prioritized experience replay() ---------------------------------------------------------------
# ==============================================================================



class SumTree(object):
    """
    This SumTree code is modified version of Morvan Zhou: 
    https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/blob/master/contents/5.2_Prioritized_Replay_DQN/RL_brain.py
    """
    data_pointer = 0
    
    """
    Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    """
    def __init__(self, capacity):
        self.capacity = capacity # Number of leaf nodes (final nodes) that contains experiences
        
        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema above
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        
        """ tree:
            0
           / \
          0   0
         / \ / \
        0  0 0  0  [Size: capacity] it's at this line that there is the priorities score (aka pi)
        """
        
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
    
    
    """
    Here we add our priority score in the sumtree leaf and add the experience in data
    """
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1
        
        """ tree:
            0
           / \
          0   0
         / \ / \
tree_index  0 0  0  We fill the leaves from left to right
        """
        
        # Update data frame
        self.data[self.data_pointer] = data
        
        # Update the leaf
        self.update (tree_index, priority)
        
        # Add 1 to data_pointer
        self.data_pointer += 1
        
        if self.data_pointer >= self.capacity:  # If we're above the capacity, you go back to first index (we overwrite)
            self.data_pointer = 0
            
    
    """
    Update the leaf priority score and propagate the change through tree
    """
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # then propagate the change through tree
        while tree_index != 0:    # this method is faster than the recursive loop in the reference code
            
            """
            Here we want to access the line above
            THE NUMBERS IN THIS TREE ARE THE INDEXES NOT THE PRIORITY VALUES
            
                0
               / \
              1   2
             / \ / \
            3  4 5  [6] 
            
            If we are in leaf at index 6, we updated the priority score
            We need then to update index 2 node
            So tree_index = (tree_index - 1) // 2
            tree_index = (6-1)//2
            tree_index = 2 (because // round the result)
            """
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    
    """
    Here we get the leaf_index, priority value of that leaf and experience associated with that index
    """
    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_index = 0
        
        while True: # the while loop is faster than the method in the reference code
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            
            else: # downward search, always search for a higher priority node
                
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                    
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
            
        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] # Returns the root node





















# ==============================================================================
# -- Store experience in memory() ---------------------------------------------------------------
# ==============================================================================



'''
class Memory():
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        self.explore_prob = 1
        self.decay_step = 0
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]
    def save(self,filename):
        with open(filename.strip("./")+".mem", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    def load(self,filename):
        with open(filename.strip("./")+".mem", "rb") as f:
            dump= pickle.load(f)
            self.explore_prob=dump.explore_prob
            self.buffer=dump.buffer
            self.decay_step=dump.decay_step
            f.close()
        
'''





class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree 
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = SumTree(capacity)
        self.explore_prob = 1
        self.decay_step = 0
    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        
        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper
        
        self.tree.add(max_priority, experience)   # set the max p for new p

        
    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        b_idx, b_ISWeights = np.empty((n,1), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b)/ max_weight                   
            b_idx[i]= index
            
            experience = [data]
            
            memory_b.append(experience)
        
        return b_idx, memory_b, b_ISWeights
    
    """
    Update the priorities on the tree
    """
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def save(self,filename):
        with open(filename.strip("./")+".mem", "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            f.close()

    def load(self,filename):
        with open(filename.strip("./")+".mem", "rb") as f:
            dump= pickle.load(f)
            self.explore_prob=dump.explore_prob
            self.tree=dump.tree
            self.decay_step=dump.decay_step
            f.close()





def pretrain_memory(args):
   # Instantiate memory
   mem = Memory(capacity = Q_network.memory_size)
   while mem.tree.data[ Q_network.pretrain_length-1] ==0:
       # Render the environment
       mem,_, _, _, _,_, _, _ =game_loop(args,memory=mem, explore_exploit_action=False, decay_step=0, episode=0, tau=0, DQNetwork=None, TargetNetwork=None, writer=None, write_op=None,  training=False, sess=None,use_model=False)
       print("Adversary: Number of experiences in pretrain memory = ", Q_network.memory_size- np.sum(mem.tree.data==0))
   mem.decay_step= 1000000   # Comment it out later 
   return mem



# ==============================================================================
# -- Deep Q netwok training() ---------------------------------------------------------------
# ==============================================================================


def trainQ_network(args, memory, data_log=None, DQNetwork=None, TargetNetwork=None, writer=None, write_op=None, training=True, TransferLearning=False, file_name=""):
    # Saver will help us to save our model
    saver = tf.train.Saver()
    if not TransferLearning:
        #file_name="./models/model"+time.asctime().replace(' ','').replace(':','_')
        file_name = "./models_adversary/model_adversary"

    with tf.Session() as sess:
        if TransferLearning:
            #initialize from pretrained model
            saver.restore(sess,file_name+".ckpt")
        else: 
            # Initialize the variables
            sess.run(tf.global_variables_initializer())  
 
        # Initialize the decay rate (that will use to reduce epsilon) 
        decay_step = 0
        tau = 0
        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = Q_network.update_target_graph()
        sess.run(update_target)
        for episode in range(Q_network.total_episodes):
            memory, decay_step, tau, DQNetwork, TargetNetwork, writer, write_op, sess=game_loop(args, data_log, memory=memory, explore_exploit_action=True, decay_step=decay_step, episode=episode, tau=tau, DQNetwork=DQNetwork, TargetNetwork=TargetNetwork, writer=writer, write_op=write_op, training=True, sess=sess, use_model=False)
            # Save model every 5 episodes
            if episode % 5 == 0 or episode == Q_network.total_episodes-1:
                save_path = saver.save(sess, file_name+".ckpt")
                memory.save(file_name)
                print("Adversary: Model Saved")
    return file_name

# ==============================================================================
# -- Execute learned model---------------------------------------------------------------
# ==============================================================================




def execute_DQN(args,DQNetwork, TargetNetwork, file_name=""):
   with tf.Session() as sess:
      saver = tf.train.Saver()
      # Load the model
      saver.restore(sess,file_name +".ckpt")
      game_loop(args, memory=None, explore_exploit_action=False, decay_step=0, episode=0,  tau=0, DQNetwork=DQNetwork, TargetNetwork=TargetNetwork, writer=None, write_op=None, training=False, sess=sess, use_model=True)
   return

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================


def game_loop(args, data_log=None, memory=None, explore_exploit_action = False, decay_step=0, episode=0, tau=0, DQNetwork=None, TargetNetwork=None, writer=None, write_op=None, training=False, sess=None, use_model=False):
    x = 1950
    y = 0
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)



    stuck = deque(maxlen = 20)
    vehicle_stuck=False
    is_new_episode=True
    pygame.init()
    pygame.display.set_caption('Adversary')
    pygame.font.init()
    world = None
    total_reward=0
    steps=0
    loss=0
    stack_size = 4 # We stack 4 frames
    # Initialize deque with zero-images one array for each image
    stacked_frames  =  deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4) 
    try:
        client = None
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)

        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(args.width, args.height)
        WORLD = client.get_world()
         #uncomment to change map
        #WORLD = client.load_world('Town04')
       # world = World(client.get_world(), hud, args.filter)

        if Q_network.TransferLearning and memory !=None:
            explore_probability=memory.explore_prob
            decay_step =memory.decay_step
            #if decay_step>1:
             #   decay_step=1000000

        
        current_map=WORLD.get_map()
        dao = GlobalRoutePlannerDAO(current_map, 2.0)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        #a = carla.Location(x=96.0,y=4.45,z=0)
        #a=world.player.get_location()
        #b = carla.Location(x=12.0, y=190, z=0)
        ROUTE = grp.trace_route(initial_point, goal_point)
        
        route=[]
        route_adversary=[]
        #current_waypoint=ROUTE[0][0].transform.rotation.yaw
        #next_waypoint=ROUTE[10][0].transform.rotation.yaw
        #my_pose=world.player.get_transform().rotation.yaw
        #print(current_waypoint, next_waypoint, my_pose)
        for g in range(len(ROUTE)-1):
            current_waypoint=ROUTE[g][0]
            route.append(current_waypoint.transform)
            route_adversary.insert(0,current_waypoint.transform)
            #print(current_waypoint.transform)
            #next_waypoint=ROUTE[g+1][0]
            #world.world.debug.draw_line(current_waypoint.transform.location, next_waypoint.transform.location, thickness=0.3, life_time= 0.2)
        world = World(WORLD, hud, args.filter, route, route_adversary)
        
        time.sleep(0.2)
        controller = DualControl(world, args.autopilot)


        args_lateral={'K_P': 1.0, 'K_D': 0.0, 'K_I': 0.3}
        args_longitudinal={'K_P':1.0, 'K_D': 1.0, 'K_I': 0.2}
        #args_lateral={'K_P': 0.0, 'K_D': 0.0, 'K_I': 0.0}
        #args_longitudinal={'K_P':0.0, 'K_D': 0.0, 'K_I': 0.0}
        pid_controller=VehiclePIDController(world.player, args_lateral,args_longitudinal)   # Initializing pid controller to speed up the training
        pid_index=0
        pure_pid =np.random.rand()


        clock = pygame.time.Clock()
        done=False
        explore_probability=1
        #Get first state
        v = world.player.get_velocity()
        a_v=world.player.get_angular_velocity()
        a= world.player.get_acceleration()
        con=world.player.get_control() 
        vehicle_transformation=world.player.get_transform().get_forward_vector()
        #print(np.linalg.norm([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z]))
        v_long=round(abs(np.dot([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z], [v.x,v.y,v.z])),2)
        v_lat=round(abs(np.linalg.norm([v.x,v.y])-v_long),2)
        a_long=round(abs(np.dot([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z], [a.x,a.y,a.z])),2)
        a_lat=round(abs(np.linalg.norm([a.x,a.y])-a_long),2)
        while most_recent_image is None:	
            pass
        next_state=most_recent_image #first state
        next_state_vehicle=np.array([v_long, v_lat,round(a_v.z,2),a_long,a_lat,round(con.throttle,2),round(con.steer,2),round(con.brake,2)])

        #print(next_state)
        next_state,stacked_frames = Q_network.stack_frames(stacked_frames, next_state, True)
        # chosen_action=Reward.choose_action()   # Random Initialization
        text=['No steer No throttle', 'steer left No throttle', 'steer right No throttle','No steer Accelerate', 'steer left Accelerate', 'steer right Accelerate','No steer Deccelerate', 'steer left Deccelerate', 'steer right Deccelerate']

        pid_speed_previous =0 
        # Check protagonist present before going in the while loop
        f = open("avoider_game_status",'r')
        ans = f.readline()
        f.close()
        done = "1" in ans
        while done:
           f = open("avoider_game_status",'r')
           ans = f.readline()
           f.close()
           done = "1" in ans
        done_protagonist=False
        Route_length =Reward.initial_dis_to_goal_adversary(world)

        while not done:
            clock.tick_busy_loop(60)
            pid_index=choose_pid_waypoint_adversary(route_adversary,world)
            v = world.player.get_velocity()
            current_state=next_state
            current_state_vehicle=next_state_vehicle
            if explore_exploit_action: #and steps%3 ==0:
                target_waypoint_transform = route_adversary[pid_index]
                player_loc=world.player.get_location()
                dist_to_target=np.linalg.norm([player_loc.x - target_waypoint_transform.location.x ,player_loc.y-target_waypoint_transform.location.y])
                target_speed=20
                #print('target_speed', target_speed)
                #print("target speed: ",target_speed,"current speed: ",3.6*math.sqrt(v.x**2 + v.y**2 + v.z**2))
                pid = pid_controller.run_step(target_speed,target_waypoint_transform)
                pid.throttle=0.6
                chosen_action, explore_probability = Q_network.predict_action(Q_network.explore_start, Q_network.explore_stop, Q_network.decay_rate, decay_step, current_state, Reward.possible_actions(), sess, DQNetwork, pid,controller._control, pure_pid=pure_pid>0.85,  vehicle_state=current_state_vehicle, use_pid_explore=True)
                choice=np.where(chosen_action==1)[0][0]
                #chosen_action, explore_probability = Q_network.predict_action(Q_network.explore_start, Q_network.explore_stop, Q_network.decay_rate, decay_step, current_state, Reward.possible_actions(), sess, DQNetwork, vehicle_state=current_state_vehicle)
                #choice=np.where(chosen_action==1)[0][0]
                #print(choice, chosen_action)
                #print(text[choice])
            #elif explore_exploit_action and steps%3 !=0:
             #   chosen_action=chosen_action   #frame_skip            
            elif use_model:
                # Take the biggest Q value (= the best action)
                state_shape=Q_network.state_size[:]
                state_shape.insert(0, 1)  # [1, state.shape]
                vehicle_state_shape=Q_network.vehicle_state_size[:]
                vehicle_state_shape.insert(0, 1)  # [1, state.shape]
                Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: current_state.reshape(state_shape), DQNetwork.vehicle_state_inputs_: current_state_vehicle.reshape(vehicle_state_shape)})
                # Take the biggest Q value (= the best action)
                choice = np.argmax(Qs)
                possible_actions=Reward.possible_actions()
                chosen_action = possible_actions[int(choice)]
            else:
                chosen_action=Reward.choose_action()

            flag=controller.parse_events(world, clock, action=chosen_action)
            if flag:     # Check if user killed the episode
                return

            world.tick(clock)
            world.render(display)

            #for g in range(len(route_adversary)-1):# len(ROUTE)-1):
            #   current_waypoint=route_adversary[g]
            #   next_waypoint=route_adversary[g+1]
            #   world.world.debug.draw_line(current_waypoint.location, next_waypoint.location, thickness=0.3, life_time= 2)



            #pygame.display.set_mode((200,200))
            pygame.display.flip()   
            
            Immediate_Reward=Reward.Calculate_Rewards_adversary(world, Route_length)
            #print ("\treward:", Immediate_Reward)
           
            #time.sleep(0.02)
            total_reward+=Immediate_Reward
            is_new_episode=False
            # Increase decay_step
            if training:
              decay_step +=1
            steps+=1
            
            stuck.append([world.player.get_control().throttle,  math.sqrt(v.x**2 + v.y**2 + v.z**2)])
            throttle_commands = np.array([each[0] for each in stuck])
            velocity = np.array([each[1] for each in stuck])
            if len(stuck)==20:
                if np.sum(throttle_commands)/len(throttle_commands)>0.6 and np.sum(velocity)/len(velocity)<0.2:
                     vehicle_stuck=True
                     print('Adversary: Vehicle_stuck',vehicle_stuck)


            goal_dis=Reward.dis_to_goal_adversary(world)
            done= len(world.collision_sensor.history)>0 or steps>Q_network.max_steps or vehicle_stuck or goal_dis<Reward.close_enough  # collision_check 
            '''
            if decay_step%3==0 and (not done):
               f = open("avoider_game_status",'r')
               ans = f.readline()
               f.close()
               done_protagonist = "1" in ans
            '''
            if done:
               next_state=np.zeros(next_state.shape)
               next_state_vehicle=np.zeros(Q_network.vehicle_state_size)
            else:
               #Get current state
               next_state=most_recent_image #current_state
               next_state, stacked_frames = Q_network.stack_frames(stacked_frames, next_state, False)
               v = world.player.get_velocity()
               a_v=world.player.get_angular_velocity()
               a= world.player.get_acceleration()
               con=world.player.get_control() 
               vehicle_transformation=world.player.get_transform().get_forward_vector()
               #print(np.linalg.norm([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z]))
               v_long=round(abs(np.dot([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z], [v.x,v.y,v.z])),2)
               v_lat=round(abs(np.linalg.norm([v.x,v.y])-v_long),2)
               a_long=round(abs(np.dot([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z], [a.x,a.y,a.z])),2)
               a_lat=round(abs(np.linalg.norm([a.x,a.y])-a_long),2)
               #tran_a_v=np.dot([vehicle_transformation.x,vehicle_transformation.y,vehicle_transformation.z], [a_v.x,a_v.y,a_v.z])
               #print('v_long',v_long, 'v_lat',v_lat)
               #print('a',a, 'tran_a',tran_a)
               #print('a_v',a_v, 'tran_a_v',tran_a_v)
               next_state_vehicle=np.array([v_long, v_lat,round(a_v.z,2),a_long,a_lat,round(con.throttle,2),round(con.steer,2),round(con.brake,2)])
               #next_state_vehicle=[v.x,v.y,a_v.z,a.x,a.y,con.throttle,con.steer,con.brake]
               #print("next_state",next_state.shape)
               #plot_frame = transform.resize(grayscale, frame_size) 
            
            if memory !=None and not use_model:
                 memory.store((current_state, chosen_action, Immediate_Reward, next_state, done, current_state_vehicle,next_state_vehicle))
                 memory.explore_prob=explore_probability
                 memory.decay_step=decay_step
                 if decay_step%3==0:# and (not done):
                     f = open("avoider_game_status",'r')
                     ans = f.readline()
                     f.close()
                     done_protagonist = "1" in ans
                     if not done:
                        done= done_protagonist  # If Protagonist dies
        
        f = open("avoider_game_status",'r')
        ans = f.readline()
        f.close()
        done_protagonist = "1" in ans
        if training and done_protagonist: # and decay_step%100==0:
            if world is not None:
                world.destroy()
                world=None
            for y in range(0, 20):
                ### LEARNING PART            
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(Q_network.batch_size)
                # Increase the C step
                tau += 1

                current_states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch]) 
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])
                current_states_vehicle_mb = np.array([each[0][5] for each in batch], ndmin=1)
                next_states_vehicle_mb = np.array([each[0][6] for each in batch], ndmin=1)

                target_Qs_batch = []

                 # Get Q values for next_state 
                Qs_next_state = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb, DQNetwork.vehicle_state_inputs_: next_states_vehicle_mb})

                # Calculate Qtarget for all actions that state
                Qs_target_next_state = sess.run(TargetNetwork.output, feed_dict = {TargetNetwork.inputs_: next_states_mb, TargetNetwork.vehicle_state_inputs_: next_states_vehicle_mb})

               
                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # We got a'
                    action_Q = np.argmax(Qs_next_state[i])
                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])
                        
                    else:
                        target = rewards_mb[i] + Q_network.gamma * Qs_target_next_state[i][action_Q]
                        target_Qs_batch.append(target)
                        

                targets_mb = np.array([each for each in target_Qs_batch])

                _, loss, absolute_errors = sess.run([DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                                    feed_dict={DQNetwork.inputs_: current_states_mb, DQNetwork.vehicle_state_inputs_: current_states_vehicle_mb,
                                               DQNetwork.target_Q: targets_mb,
                                               DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})

                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                
                
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: current_states_mb, DQNetwork.vehicle_state_inputs_: current_states_vehicle_mb,
                                                   DQNetwork.target_Q: targets_mb,
                                                   DQNetwork.actions_: actions_mb,
                                              DQNetwork.ISWeights_: ISWeights_mb})
                writer.add_summary(summary, episode)
                writer.flush()
                print('Adversary: Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
                data_log.write('%f, %f, %f' % (total_reward, loss, explore_probability))         
                data_log.write('\n')
                if tau > Q_network.max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = Q_network.update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Adversary: Model updated")



        if use_model:
              print('Adversary: Congratulations !!! Successfully used trained model')
              print('Adversary: Total reward: {}'.format(total_reward))
        '''
        if training:
              print('Adversary: Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability))
              data_log.write('%f, %f, %f' % (total_reward, loss, explore_probability ))         
              data_log.write('\n')
        '''
        pygame.display.quit
        return memory, decay_step, tau, DQNetwork,TargetNetwork, writer, write_op, sess

    finally:
        if world is not None:
            world.destroy()
        #pass
        #pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
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
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1800x1020',     #'3770x1020' 1280x720
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.ford*',
        help='actor filter (default: "vehicle.*")')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        
        # Introducing G29 Steering Force Feedback   NOT REQUIRED FOR ADVERSARIAL REINFORCEMENT LEARNING PROJECT
        device = evdev.list_devices()[0]
        evtdev = InputDevice(device)
        val = 54000 # val \in [0,65535]
        evtdev.write(ecodes.EV_FF, ecodes.FF_AUTOCENTER, val)



        #test_environment(args)
        #game_loop(args)

        # Reset the graph
        tf.reset_default_graph()
        tf.compat.v1.reset_default_graph
        # Instantiate the DQNetwork
        state_size=Q_network.state_size[:]
        DQNetwork = Q_network.DDDQNNet(state_size, Q_network.action_size, Q_network.learning_rate, name="DQNetwork")
        # Instantiate the target network
        TargetNetwork = Q_network.DDDQNNet(state_size, Q_network.action_size, Q_network.learning_rate, name="TargetNetwork")
        file_name = "./models_adversary/model_adversary"

                 
        # If training agent
        if Q_network.Train_network_adversary:
            if Q_network.TransferLearning_adversary:
                memory=Memory(capacity=Q_network.memory_size)
                try: 
                    memory.load(file_name)
                    print("Adversary: memory decay_step after loading",memory.decay_step, memory.explore_prob)
                except:
                    print('Adversary: No Pretrain Memory Exist. Initializing Memory!!!')
                    memory =pretrain_memory(args)
                    memory.save(file_name)
            else:
                print('Adversary: Initializing Memory!!!')
                memory =pretrain_memory(args)
                memory.save(file_name)
            print('Adversary: Memory Experiences Initialized. Starting to train the network!!!')
            writer = tf.summary.FileWriter("/tensorboard/dddqn/1")
            ## Losses
            tf.summary.scalar("Loss", DQNetwork.loss)
            write_op = tf.summary.merge_all()
            path = 'data_adversary.txt'
            data_log = open(path,'a+') # Reward, Loss Exploration, probability, mean time to failure
            file_name=trainQ_network(args, memory,data_log, DQNetwork, TargetNetwork,  writer, write_op, training=Q_network.Train_network, TransferLearning=Q_network.TransferLearning_adversary,file_name=file_name)
            print('Adversary: Network training complete!!! Lets play')
            data_log.close()
        if Q_network.use_model:
           for x in range(0, 15): 
               execute_DQN(args,DQNetwork, TargetNetwork, file_name)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()
