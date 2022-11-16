#!/usr/bin/env python


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')   
except IndexError:
    pass

import carla

import math
import random


def get_transform(vehicle_location, angle, d=6.4):
    a = math.radians(angle)
    location = carla.Location(d * math.cos(a), d * math.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-15))


def main():
    client = carla.Client('127.0.0.1', 2000)
    #client = carla.Client('localhost', 2000)
    #print('why')
    client.set_timeout(2.0)
    world = client.get_world()
    spectator = world.get_spectator()
    all_actors= world.world.get_actors()
    all_vehicles = all_actors.filter('vehicle.tesla*') 
    if len(all_vehicles)>1:
       print("multiple vehicle")
    else:
        protagonist = all_vehicles[0]
        spectator.set_transform(get_transform(protagonist.get_location(), yaw = protagonist.get_rotation().yaw))
       
if __name__ == '__main__':

    main()
