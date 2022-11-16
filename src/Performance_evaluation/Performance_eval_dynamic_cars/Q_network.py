
import tensorflow as tf       # Deep Learning library
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np            # Handle matrices


import random                 # Handling random number generation
import time                   # Handling time calculation
from skimage import transform # Helps in preprocessing the frames
import skimage
from collections import deque # Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs

import warnings # This ignore all the warning messages that are normally printed during the training because of skiimage
warnings.filterwarnings('ignore')

import Reward

import Simulator

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

frame_size=[100,120]
def simplify_semantic_image(frame):   
    simplified = np.array(frame,copy=True)  
    vegetation_color =[107,142,35]
    traffic_sign_color =[220,220,0]
    roadline_color =[157,234,50]
    road_color = [128,64,128]
    #roadline_color1 = [155,232,0]
    #roadline_color2 = [159,236,2]
    mask1 = (frame == vegetation_color).all(axis=2)
    mask2 = (frame == traffic_sign_color).all(axis=2)
    mask3 = (frame == roadline_color).all(axis=2)    
  #  print((frame >= roadline_color1).all() and (frame <= roadline_color2).all())
    #mask4 = ((frame >= roadline_color1).all() and (frame <= roadline_color2).all()) #.all(axis=2) 
    simplified[mask1]=[0,0,0]
    simplified[mask2]=[0,0,0]
    simplified[mask3]=road_color
    #simplified[mask4]=road_color
    return simplified

def preprocess_frame(frame):     
    #frame1=simplify_semantic_image(frame)
    #plt.imshow(frame1)
    #plt.show()
    grayscale=np.dot(frame, [0.2989, 0.5870, 0.1140])
    #grayscale = skimage.color.rgb2gray(frame)
    # Crop the screen (remove the roof because it contains no information)
    #cropped_frame = frame[30:-10,30:-30]
    
    # Normalize Pixel Values
    normalized_frame = grayscale/255.0
    
    # Resize
    preprocessed_frame = transform.resize(normalized_frame, frame_size)    # Earlier [84, 84]     
    #plot_frame = transform.resize(grayscale, frame_size) 
    #plt.imshow(plot_frame, cmap='gray', vmin=0, vmax=255)
    #plt.show()
    return preprocessed_frame



stack_size = 4 # We stack 4 frames
# Initialize deque with zero-images one array for each image
#stacked_frames  =  deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4) 

def stack_frames(stacked_frames, state, is_new_episode):
    # Preprocess frame
    frame = preprocess_frame(state)

    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((100,120), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)
        #plt.imshow(frame, cmap='gray', vmin=0, vmax=1)
        #plt.show()
        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 

    return stacked_state, stacked_frames




### MODEL HYPERPARAMETERS
state_size = [100,120,4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels) 
vehicle_state_size=[8]
action_size = len(Reward.possible_actions())              # 9 possible actions
learning_rate =  0.0002      # Alpha (aka learning rate)

### TRAINING HYPERPARAMETERS
total_episodes =  100000  #10000       # Total episodes for training
max_steps = 1000000              # Max possible steps in an episode
batch_size = 1000            


# FIXED Q TARGETS HYPERPARAMETERS 
max_tau = 9 #10000 #Tau is the C step where we update our target network

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.000001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95               # Discounting rate

### MEMORY HYPERPARAMETERS
pretrain_length = 10000   # Number of experiences stored in the Memory when initialized for the first time   Earlier batch_size
memory_size = 10000          # Number of experiences the Memory can keep


## TURN THIS TO TRUE IF ADDING ADVERSARY
Adversary_present = True

## TURN THIS TO TRUE FOR TRAINING
Train_network = False
## TURN THIS TO TRUE FOR TRAINING Adversary
Train_network_adversary = False


## TURN THIS TO TRUE IF YOU WANT TO USE THE TRAINED MODEL
use_model = True
use_model_adversary = True
## TURN THIS TO TRUE IF YOU WANT TO START TRAINING FROM AN EXISTING MODEL
TransferLearning=True
## TURN THIS TO TRUE IF YOU WANT TO START TRAINING FROM AN EXISTING MODEL
TransferLearning_adversary=True

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True

## TURN THIS TO TRUE IF YOU WANT TO START WITH DECAY_STEP=0
restart_training=True


class DDDQNNet:
 
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
      
        
        
        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.compat.v1.variable_scope(self.name):
            
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            placeholder_state=state_size[:]
            placeholder_state.insert(0, None)  # [None, 100, 120, 4]
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, placeholder_state, name="inputs")
            placeholder_vehicle_state=vehicle_state_size[:]
            placeholder_vehicle_state.insert(0, None)  # [None, 100, 120, 4]
            self.vehicle_state_inputs_ = tf.compat.v1.placeholder(tf.float32, placeholder_vehicle_state, name="vehicle_state_inputs")
            #
            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None,1], name='IS_weights')
            
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, action_size], name="actions_")
            
            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")
            
            """
            First convnet:
            CNN
            ELU
            """
            # Input is 100x120x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 32,
                                         kernel_size = [8,8],
                                         strides = [4,4],
                                         padding = "VALID",
                                         name = "conv1")
            

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            #print('Conv 1 elu dimention=',self.conv1_out.get_shape())
            
            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                 filters = 64,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 name = "conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")
            #print('Conv 2 elu dimention=',self.conv2_out.get_shape())
            
            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                 filters = 128,
                                 kernel_size = [4,4],
                                 strides = [2,2],
                                 padding = "VALID",
                                 name = "conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            #print('Conv 3 elu dimention=',self.conv3_out.get_shape())
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            #print('Flatten dimention=',self.flatten.get_shape())
            
            self.vehicle_states = tf.layers.dense(inputs = self.vehicle_state_inputs_,
                                  units = 8,
                                  activation = tf.nn.elu,
                                  name="vehicle_states")
            #print('vehicle_states dimention=',self.vehicle_states_inputs.get_shape())
            self.combined=tf.concat([self.flatten, self.vehicle_states],1)
            #print('combined dimention=',self.combined.get_shape())
            ## Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs = self.combined,
                                  units = 512,
                                  activation = tf.nn.elu,
                                  name="value_fc")

            #print('Value fc dimention=',self.value_fc.get_shape())
            self.value = tf.layers.dense(inputs = self.value_fc,
                                        units = 1,
                                        activation = None,
                                        name="value")
            
            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs = self.combined,
                                  units = 512,
                                  activation = tf.nn.elu,
                                name="advantage_fc")
            
            self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                        units = self.action_size,
                                        activation = None,
                                name="advantages")

            
            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))                
            #print('self.output dimention=',self.output.get_shape())
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)
            #print('self.Q dimention=',self.Q.get_shape())
            # The loss is modified because of PER 
            self.absolute_errors = tf.abs(self.target_Q - self.Q)# for updating Sumtree
            
            self.loss = tf.reduce_mean(self.ISWeights_ * tf.squared_difference(self.target_Q, self.Q))
            
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)






         

"""
This function will do the part
With epsilon select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(explore_start, explore_stop, decay_rate, decay_step, state, actions, sess, DQNetwork, pid_control =None,current_control=None, pure_pid=False, vehicle_state=None, use_pid_explore=False):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    exp_exp_tradeoff = np.random.rand()
    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff or pure_pid):   # Exploration
        if use_pid_explore:
            pid_explore = np.random.rand()
        else: 
            pid_explore = 0 #np.random.rand()
        if (pid_explore> 1-explore_probability*0.70 or pure_pid) and pid_control !=None:#(1-explore_probability*0.75) and pid_control !=None:
            action=Reward.pid_control_action(pid_control,current_control)
            #print("Choosing PID action")
        else:
            # Make a random action (exploration)
            action = random.choice(Reward.possible_actions())
            #print("Choosing random action")
        
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state

        state_shape=state_size[:]
        state_shape.insert(0, 1)  # [1, state.shape]
        vehicle_state_shape=vehicle_state_size[:]
        vehicle_state_shape.insert(0, 1)  # [1, state.shape]
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape(state_shape), DQNetwork.vehicle_state_inputs_: vehicle_state.reshape(vehicle_state_shape)})
        possible_actions= Reward.possible_actions()
        # Take the biggest Q value (= the best action)
        choice = np.argmax(Qs)
        action = possible_actions[int(choice)]
        #print("Choosing learned action")
    return action, explore_probability



# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani
def update_target_graph():
    
    # Get the parameters of our DQNNetwork
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
    
    # Get the parameters of our Target_network
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []
    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder




def main():
    return



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass

