import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import argparse
import logging

import gym
import math

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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

################################### Training ###################################
def train(args):
    print("============================================================================================")
    ####### initialize environment hyperparameters ######
    env_name = 'gym_carla:carla-v0'
    Are_actions_continuous = True
    nbObstacles = 0                     # Be careful the size of the obstacle might affect 
                                        # its detection by the vehicle

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = 100000     # break training loop if timeteps > max_training_timesteps
    #####################################################

    print("training environment name : " + env_name)
    env = gym.make(env_name,args=args, render_mode="human",nbObstacles=0,cont_act=Are_actions_continuous)
    check_env(env)
    model = PPO("MultiInputPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=max_training_timesteps)
    model.save("ppo_carla")
    print("============================================================================================")
    print("Training complete!")
    print("============================================================================================")
    model = PPO.load("ppo_carla")
    # test
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        #env.render(mode='rgb_array')



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
        '-t', '--town',
        metavar='TOWN',
        help='load a default map from CARLA')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.add_argument(
        '-s', '--server-path',
        metavar='SERVER_FILE_PATH',
        help='path to server')
    args = argparser.parse_args()

    logging.info('listening to server %s:%s', args.host, args.port)

    #print(__doc__)

    try:

        train(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()