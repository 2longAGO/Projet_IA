"""
This file exists to test a pretrained PPO agent
Requires CARLA to function 
When closing close the client first!!!
AFTER you can close the server if you please
"""
"""
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
"""

import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import argparse
import logging

import gymnasium as gym
import math
import matplotlib as mpl
import matplotlib.pyplot as plt

from PPO import PPO

#################################### Testing ###################################
def test(args):
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = 'gym_carla:carla-v0'
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

    print("training environment name : " + env_name)

    env = gym.make(env_name,args=args, render_mode="human")
    state, *_ = env.reset()
    #print(env.observation_space.shape)
    # state space dimension
    state_dim = state['obstacles'].size+2 #env.observation_space.shape

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num
    env_file = env_name.replace(':', '_',-1) ### env related pretrained model file name

    directory = "PPO_preTrained" + '/' + env_file + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_file, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)
    if os.path.exists(checkpoint_path) :
        ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0
    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        state, *_ = env.reset()

        for t in range(1, max_ep_len+1):
            processed_state = np.append(state['obstacles'],[state['speed'],state['distTarget']])

            # select action with policy
            action = ppo_agent.select_action(processed_state)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if render:
                env.render()
                #time.sleep(frame_delay)

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

    print(__doc__)

    try:

        test(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()