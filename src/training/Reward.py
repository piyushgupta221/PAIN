
import numpy as np 
import carla
import math
import random

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


## Global variables

brake_penalty=3
max_velocity= 7.0
max_acceleration= 4.5
reward_velocity=1
reward_acc=1
stop_penalty=3
steer_penalty=2
goal_reward=5
goal_reward_adversary=2
At_goal_reward=100
cross_error_penalty=3
close_to_agent_reward=7
At_goal_reward_adversary=10
close_enough = 5
cross_error_penalty_adversary=2

episodes=20
def possible_actions():
    
    # Possible actions
    continue_forward=np.zeros(9, dtype=int)    # No steer No change in throttle 
    continue_left=np.zeros(9, dtype=int)      # steer -2deg No change in throttle 
    continue_right=np.zeros(9, dtype=int)    # steer +2deg No change in throttle 
    accel_forward=np.zeros(9, dtype=int)     # No steer throttle +0.2 
    accel_left=np.zeros(9, dtype=int)   # steer -2deg throttle +0.2 
    accel_right=np.zeros(9, dtype=int)   # steer +2deg throttle +0.2 
    decel_forward=np.zeros(9, dtype=int)  # No steer throttle -0.2 
    decel_left=np.zeros(9, dtype=int)   # steer -2deg throttle -0.2 
    decel_right=np.zeros(9, dtype=int)   # steer +2deg throttle -0.2 
    continue_forward[0] =1
    continue_left[1] =1
    continue_right[2] =1
    accel_forward[3] =1
    accel_left[4] =1
    accel_right[5] =1
    decel_forward[6] =1
    decel_left[7] =1
    decel_right[8] =1
    available_actions = [continue_forward, continue_left, continue_right, accel_forward, accel_left, accel_right, decel_forward, decel_left, decel_right]
    
    return available_actions



def choose_action():
    available_actions=possible_actions()
    #action=available_actions[3]
    action = random.choice(available_actions)
    return action


def choose_control(action, current_control):
    #print('current_throttle=', current_control.throttle, 'current_brake=', current_control.brake)
    prev_throttle=current_control.throttle
    prev_brake=current_control.brake
    new_control=current_control
    step =0.1
    new_control.throttle= round(min(max((new_control.throttle + np.dot(action,np.multiply(step,[0, 0, 0, 1, 1, 1, -1, -1, -1]))),0), 1),2)  # Calculating throttle command
    new_control.steer= round(min(max((new_control.steer + np.dot(action,np.multiply(step,[0, -1, 1, 0, -1, 1, 0, -1, 1]))),-1),1),2)  # Calculating Steering command
    if round(new_control.throttle,2)>0 or round(prev_throttle,2)>0:
       new_control.brake=0
    else:
       new_control.brake= round(min(max((new_control.brake + np.dot(action,np.multiply(step,[0, 0, 0, -1, -1, -1, 1, 1, 1]))),0),1),2) # Calculating Braking Command

    return new_control


def pid_control_action(pid_control, current_control):
    step_size =0.05
    if pid_control.throttle-current_control.throttle>step_size:#up
       accel=2  # action 3,4,5
    elif abs(pid_control.throttle-current_control.throttle)<=step_size:#stay
       accel=1   # action 0,1,2
    else:#down
       accel=0  # action 6,7,8
    if pid_control.steer- current_control.steer>step_size:#right
       steer=2  # 2, 5, 8
    elif abs(pid_control.steer- current_control.steer)<=step_size:#stay
       steer=1  # 0, 3, 6
    else: #left
       steer=0  #1  4 7
    available_actions=possible_actions()
    action_space = [[7,6,8],[1,0,2],[4,3,5]]
    return available_actions[action_space[accel][steer]]



def closestRoutePoint(world,start_index=0):
    player_loc=world.player.get_location()
    ix=start_index
    dis_prev=1000000
    for index in range(len(world.Route)-1):
        dis_current=np.linalg.norm([player_loc.x - world.Route[index].location.x , player_loc.y-world.Route[index].location.y])
       
        if dis_current<dis_prev:
            ix=index
            dis_prev=dis_current
    #print(dis_current,ix)
    return world.Route[ix], ix


def closestRoutePoint_adversary(world,start_index=0):
    player_loc=world.player.get_location()
    ix=start_index
    dis_prev=1000000
    for index in range(len(world.Route_adversary)-1):
        dis_current=np.linalg.norm([player_loc.x - world.Route_adversary[index].location.x , player_loc.y-world.Route_adversary[index].location.y])
       
        if dis_current<dis_prev:
            ix=index
            dis_prev=dis_current
    #print(dis_current,ix)
    return world.Route_adversary[ix], ix

def closestRoutePoint2(world,start_index=0):
    player_loc=world.player.get_location()
    dis=100
    finish=False
    ix=start_index
    dis_prev=1000000
    while not finish:
        dis_current=np.linalg.norm([player_loc.x - world.Route[ix].location.x , player_loc.y-world.Route[ix].location.y])
        #print(dis_current,ix)
        if dis_current>dis_prev:
            finish=True
        else:
            dis_prev=dis_current
        ix=ix+1
        if ix>len(world.Route):
            finish=True
    return world.Route[ix-1], ix-1

def Calculate_Rewards(world, Route_length, segment_length=0,n=30):
    my_location= world.player.get_location()
    current_map=world.world.get_map()
    closest_waypoint, _= closestRoutePoint(world)#current_map.get_waypoint(my_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    cross_track_error=np.linalg.norm([my_location.x - closest_waypoint.location.x , my_location.y-closest_waypoint.location.y])
    #print(cross_track_error)
    my_velocity= world.player.get_velocity()
    #my_angular_velocity= world.player.get_angular_velocity()
    my_acceleration= world.player.get_acceleration()
    velocity= math.sqrt(my_velocity.x**2 + my_velocity.y**2 + my_velocity.z**2)
    acceleration= math.sqrt(my_acceleration.x**2 + my_acceleration.y**2 + my_acceleration.z**2)
    road_heading = closest_waypoint.rotation.yaw
    cos_alpha=math.cos((road_heading-world.player.get_transform().rotation.yaw)*math.pi/180)#(road_heading.x*my_velocity.x + road_heading.y*my_velocity.y + road_heading.z*my_velocity.z)/(mag_road_direction*velocity)
   # print(cos_alpha)
    sin_alpha=math.sqrt(1-cos_alpha**2)
    collide=world.collision_sensor.actor_collision_list
    collision_reward=0
    if len(collide)>0:
        collision_reward=-25
    #print(velocity,max_velocity)
    steering=world.player.get_control().steer
    brake=world.player.get_control().brake
    goal_dis=dis_to_goal(world)
    maximum_velocity=max_velocity+5.0
    #print('brake',brake)
    Imediate_Reward= round(reward_velocity*(velocity/maximum_velocity)*(cos_alpha-abs(sin_alpha))*(velocity <= maximum_velocity) -reward_acc*(acceleration> max_acceleration)- stop_penalty*(velocity<=1) -steer_penalty*steering**2 + collision_reward + goal_reward*(1-goal_dis/Route_length)
                                                 + At_goal_reward*(goal_dis<close_enough) -cross_error_penalty*(cross_track_error**2/40)*(abs(cross_track_error)>2) - round(brake_penalty*abs(brake),2),2)# Use this for penalizing lateral velocity

    return Imediate_Reward

def dis_to_goal(world):
    closest_waypoint, ix= closestRoutePoint(world)
    #print(ix)
    dis_goal=0
    while ix<len(world.Route)-1:
        dis = np.linalg.norm([world.Route[ix+1].location.x - world.Route[ix].location.x , world.Route[ix+1].location.y-world.Route[ix].location.y])
        dis_goal=dis_goal+dis
        ix=ix+1
    return dis_goal

def dis_to_goal_adversary(world):
    closest_waypoint, ix= closestRoutePoint_adversary(world)
    #print(ix)
    dis_goal=0
    while ix<len(world.Route_adversary)-1:
        dis = np.linalg.norm([world.Route_adversary[ix+1].location.x - world.Route_adversary[ix].location.x , world.Route_adversary[ix+1].location.y-world.Route_adversary[ix].location.y])
        dis_goal=dis_goal+dis
        ix=ix+1
    return dis_goal

def dis_to_subgoal(world,n=30):
    #subgoals are Route[30 60 90 ... 660] for n = 30 
    closest_waypoint, ix= closestRoutePoint(world)
    sub_goal_ix=min(int(n*math.floor((ix+1)/n)+n),len(world.Route))
    dis_sub_goal=2.0*(sub_goal_ix-ix)+np.linalg.norm([world.Route[ix+1].location.x - world.Route[ix].location.x , world.Route[ix+1].location.y-world.Route[ix].location.y])
    return dis_sub_goal

def initial_dis_to_goal(world):
    ix=0
    #print(ix)
    dis_goal=0
    while ix<len(world.Route)-1:
        dis = np.linalg.norm([world.Route[ix+1].location.x - world.Route[ix].location.x , world.Route[ix+1].location.y-world.Route[ix].location.y])
        dis_goal=dis_goal+dis
        ix=ix+1
    return dis_goal

def initial_dis_to_goal_adversary(world):
    ix=0
    #print(ix)
    dis_goal=0
    while ix<len(world.Route_adversary)-1:
        dis = np.linalg.norm([world.Route_adversary[ix+1].location.x - world.Route_adversary[ix].location.x , world.Route_adversary[ix+1].location.y-world.Route_adversary[ix].location.y])
        dis_goal=dis_goal+dis
        ix=ix+1
    return dis_goal



def segment_len(world,n=30):
    ix=0
    #print(ix)
    dis_seg=0
    while ix<n-1:
        dis = np.linalg.norm([world.Route[ix+1].location.x - world.Route[ix].location.x , world.Route[ix+1].location.y-world.Route[ix].location.y])
        dis_seg=dis_seg+dis
        ix=ix+1
    return dis_seg


def Calculate_Rewards_adversary(world, Route_length):

    my_location= world.player.get_location()
    current_map=world.world.get_map()
    closest_waypoint, _= closestRoutePoint_adversary(world)#current_map.get_waypoint(my_location, project_to_road=True, lane_type=carla.LaneType.Driving)
    cross_track_error=np.linalg.norm([my_location.x - closest_waypoint.location.x , my_location.y-closest_waypoint.location.y])
    my_velocity= world.player.get_velocity()
    #my_angular_velocity= world.player.get_angular_velocity()
    my_acceleration= world.player.get_acceleration()
    velocity= math.sqrt(my_velocity.x**2 + my_velocity.y**2 + my_velocity.z**2)
    acceleration= math.sqrt(my_acceleration.x**2 + my_acceleration.y**2 + my_acceleration.z**2)
    road_heading = closest_waypoint.rotation.yaw -180  # Reversing the road heading because route for adversary derived by reversing route of protagonist
    #print(road_heading,world.player.get_transform().rotation.yaw)
    cos_alpha=math.cos((road_heading-world.player.get_transform().rotation.yaw)*math.pi/180)#(road_heading.x*my_velocity.x + road_heading.y*my_velocity.y + road_heading.z*my_velocity.z)/(mag_road_direction*velocity)
    #print(cos_alpha)
    # print(cos_alpha)
    sin_alpha=math.sqrt(1-cos_alpha**2)
    collision_reward=0
    collide=world.collision_sensor.actor_collision_list
    if len(collide)>0:
        for c in collide:
            if "vehicle" in c.type_id:
                collision_reward =1000
                break
            else:
                collision_reward=-40
    dist = 1e6
    all_actors= world.world.get_actors()
    all_vehicles = all_actors.filter('vehicle.*') 
    for actor in all_vehicles:
        if world.player.id != actor.id:
            dist = np.linalg.norm([my_location.x - actor.get_location().x , my_location.y-actor.get_location().y])
          
    steering=world.player.get_control().steer
    goal_dis=dis_to_goal_adversary(world)
    #Imediate_Reward= reward_par1*velocity*(abs(cos_alpha) - abs(sin_alpha) )*(velocity <= max_velocity) -reward_par2*(acceleration> max_acceleration)      # Use this for penalizing lateral velocity
    Imediate_Reward=  round(reward_velocity*(velocity/max_velocity)*(cos_alpha-abs(sin_alpha)) -steer_penalty*steering**2  + collision_reward + close_to_agent_reward*(1-min(1,dist/100)),2)+ goal_reward_adversary*(1-goal_dis/Route_length)-cross_error_penalty_adversary*(cross_track_error**2/40)*(abs(cross_track_error)>2)+ At_goal_reward_adversary*(goal_dis<close_enough)
    return Imediate_Reward


def main():
    return
    




if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass

