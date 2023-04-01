import glob
import os
import sys
import subprocess
"""
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
"""

import carla
import gymnasium as gym

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import numpy as np

#import pygame
#from pygame.locals import*

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loop(args):
    env = gym.make('gym_carla:carla-v0',args=args, render_mode="human")
    env.reset()
    test = True
    while True:
        #if test:
        #    env.render()
        #    test=False
        #pass
        env.render()
    pass

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

    print(__doc__)

    try:

        loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()


# Might be useful
#
#            if info["closed"]: # Check if closed
#                exit(0)