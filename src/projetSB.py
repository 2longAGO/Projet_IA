import torch
import numpy as np

import gym
import math

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from drivers.disparity import DisparityExtender
from drivers.follow_the_gap import GapFollower
from drivers.starting_point import SimpleDriver

velVector = lambda x,y: math.sqrt(x**2+y**2)
# Change number of rays 
def preprocess_lidar(ranges,nbRays=16):
    """ Any preprocessing of the LiDAR data can be done in this function.
        Possible Improvements: smoothing of outliers in the data and placing
        a cap on the maximum distance a point can be.
    """
    # remove quadrant of LiDAR directly behind us
    # print(type(ranges))
    eighth = int(len(ranges) / 8)
    buf_ranges = ranges[eighth:-eighth]
    return np.array(buf_ranges[range(0,len(buf_ranges),(len(buf_ranges)//nbRays) if nbRays > 0 else 1)])

def reward_fn(state,reward):
    # state contains
    # Linear_vels_x Linear_vels_y current speed of each vehicle on the track
    # collisions of each vehicle on the track
    # poses_x poses_y current position of each vehicle on the track
    # lap_counts number of laps of the circuit
    # lap_times time taken for a lap and current time of the lap
    # We only have 1 vehicle so we get the 0 for the speed of the singular vehicle
    """
        "ang_vels_z": spaces.Box(low=-np.inf, high=np.inf, shape=(num_agents,),dtype=np.float32),
        "collisions": spaces.Box(low=0, high=1, shape=(num_agents,),dtype=np.float32),
        "ego_idx": spaces.Discrete(self.num_agents),
        "lap_counts": spaces.Box(low=0, high=np.inf, shape=(num_agents,),dtype=np.int),
        "lap_times": spaces.Box(low=0, high=np.inf, shape=(num_agents,),dtype=np.float32),
        "Linear_vels_x": spaces.Box(low=0, high=np.inf, shape=(num_agents,),dtype=np.float32),
        "Linear_vels_y": spaces.Box(low=0, high=np.inf, shape=(num_agents,),dtype=np.float32),
        "poses_theta": spaces.Box(low=0, high=np.inf, shape=(num_agents,),dtype=np.int),
        "poses_x": spaces.Box(low=-np.inf, high=np.inf, shape=(num_agents,),dtype=np.float32),
        "poses_y": spaces.Box(low=-np.inf, high=np.inf, shape=(num_agents,),dtype=np.float32),
    """
    # transform the reward based on the current speed of the vehicle
    reward += min(velVector(state['linear_vels_x'][0],state['linear_vels_y'][0]),1)
    # Give reward for proximity to DisparityExtender
    reward += 2/max(velVector(state['poses_x'][0] - state['poses_x'][1],state['poses_y'][0]-state['poses_y'][1]),2)
    # reduce reward if a collision happens
    reward -= 75 if state['collisions'].any() == 1.0 else 0
    # (TO-DO) Add incentive to go forward
    
    return reward

################################### Testing ###################################
def test(): # args
    #
    ################################## set device ##################################
    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = 'f110-v0'
    racetrack = "TRACK_2"
    listDrivers = [GapFollower()] # SimpleDriver(),DisparityExtender()
    render = True
    file_name = "ppo_f110_3"
    #####################################################

    print("testing environment name : " + env_name)
    env = gym.make('f110_gym:f110-v0', map="maps/{}".format(racetrack), map_ext=".png", render_step=render, reward_fn=reward_fn, lidar_fn=preprocess_lidar, drivers=listDrivers)
    check_env(env)
    model = PPO.load(file_name, device=device)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
    print("============================================================================================")


if __name__ == '__main__':

    test()