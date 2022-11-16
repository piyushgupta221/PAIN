import glob
# Just disables the warning, doesn't enable AVX/FMA
import os
import sys
import time
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

port = 2000
map = "Town04"

args = sys.argv
for arg in range(len(args)):
    if args[arg]=="map":
	    map = args[arg+1]
    if args[arg]=="port":
	    port = args[arg+1]
if len(args) == 2:
    map = args[1]
cli = carla.Client('localhost',port)
cli.set_timeout(2.0)
cli.load_world(map)


os.system('sudo python Simulator.py &') 
#os.system('sudo python SpectatorFolow.py &') 
#time.sleep(1)
#os.system('sudo python spawn_npc.py -n 10 -w 0 &')
os.system('sudo python Simulator_Adversary.py &')
