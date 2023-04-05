import os
import glob
import time
from datetime import datetime

import torch
import yaml
import numpy as np
from argparse import Namespace

import gym
import math

from PPO import PPO
from LidarVis import Visualiser, calc_end_pos

velVector = lambda x,y: math.sqrt(x**2+y**2)

def preprocess_lidar(ranges):
    """ Any preprocessing of the LiDAR data can be done in this function.
        Possible Improvements: smoothing of outliers in the data and placing
        a cap on the maximum distance a point can be.
    """
    # remove quadrant of LiDAR directly behind us
    eighth = int(len(ranges) / 8)
    return np.array(ranges[eighth:-eighth])

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = 'f110-v0' # "RoboschoolWalker2d-v1" 
    has_continuous_action_space = True 
    max_ep_len = 1000           # max timesteps in one episode
    action_std = 0.1            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 80               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.0003           # learning rate for actor
    lr_critic = 0.001           # learning rate for critic

    #####################################################
    #vehicle_idx # might be used
    def race(drivers=[1],
            racetrack='TRACK_1',
            vis_driver_idx=0,
            visualise_lidar=True):
        """
        :param racetrack: (TRACK_1, TRACK_2, TRACK_3, OBSTACLES)
        :param drivers: A list of classes with a process_lidar
        function.
        """
        with open('maps/{}.yaml'.format(racetrack)) as map_conf_file:
            map_conf = yaml.load(map_conf_file, Loader=yaml.FullLoader)
        scale = map_conf['resolution'] / map_conf['default_resolution']
        starting_angle = map_conf['starting_angle']
        return gym.make('f110_gym:f110-v0', map="maps/{}".format(racetrack),
                map_ext=".png", num_agents=len(drivers), disable_env_checker = True), \
                np.array([[-1.25*scale + (i * 0.75*scale), 0., starting_angle] for i in range(1)]) , \
                visualise_lidar

    env, poses, visualise_lidar = race()
    # specify starting positions of each agent
    if visualise_lidar:
        vis = Visualiser()
    obs, step_reward, done, info = env.reset(poses=poses)

    # state space dimension
    state_dim = preprocess_lidar(obs['scans'][0]).size + 1#len(obs)-2 #, info = env.reset()#env.observation_space.shape[0]
    #print(state_dim)

    # action space dimension
    if has_continuous_action_space:
        # print(spaces.Box(np.array[-2,0],np.array[2,10],dtype=np.float32))
        # spaces.Box(np.array[-2,0],np.array[2,10],dtype=np.float32).shape[0]
        action_dim = 2
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)
    if os.path.exists(checkpoint_path) :
        ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, *_ = env.reset(poses=poses)

        for t in range(1, max_ep_len+1):
            actions = []
            processed_state = np.append(preprocess_lidar(state['scans'][0]),[velVector(state['linear_vels_x'][0],state['linear_vels_y'][0])])
            print('speed: ',velVector(state['linear_vels_x'][0],state['linear_vels_y'][0]))
            action = ppo_agent.select_action(processed_state)
            actions.append(list(action))
            actions = np.array(actions)
            state, reward, done, _ = env.step(actions)
            ep_reward += reward*velVector(state['linear_vels_x'][0],state['linear_vels_y'][0])*0.5

            if render:
                proc_ranges = obs['scans'][0]
                vis.step(proc_ranges)
                env.render(mode='human_fast') # human to make it easier to discern
                #time.sleep(frame_delay)

            if state['collisions'].any() == 1.0:
                ep_reward -= 10
                break

            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()