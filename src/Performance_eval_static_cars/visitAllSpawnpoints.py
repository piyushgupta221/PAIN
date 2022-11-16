#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
# Just disables the warning, doesn't enable AVX/FMA
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


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


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
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle')
    sp =world.get_map().get_spawn_points()
    location = random.choice(sp).location
    print(len(sp))

    for point in sp:
        transform = carla.Transform(point.location, carla.Rotation(yaw=-45.0))
        vehicle = world.spawn_actor(vehicle_blueprints[0], transform)
        print(transform)
        try:
            angle = 0
            while angle < 45:
                timestamp = world.wait_for_tick()
                angle += timestamp.delta_seconds * 60.0
                spectator.set_transform(get_transform(vehicle.get_location(), angle - 90))

        finally:

            vehicle.destroy()
if __name__ == '__main__':

    main()
