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
render = True
# Change number of rays 
def preprocess_lidar(ranges,nbRays=16):
    """ Any preprocessing of the LiDAR data can be done in this function.
        Possible Improvements: smoothing of outliers in the data and placing
        a cap on the maximum distance a point can be.
    """
    # remove quadrant of LiDAR directly behind us
    # print(type(ranges))
    #eighth = int(len(ranges) / 8)
    #buf_ranges = ranges  #[eighth:-eighth]
    #return np.array(buf_ranges[range(0,len(buf_ranges),(len(buf_ranges)//nbRays) if nbRays > 0 else 1)])
    stop = min(945,len(ranges))
    return np.array(ranges[range(135,stop,(stop//nbRays+2) if nbRays > 0 else 1)])

def reward_fn(state,reward):
    # state contains
    # Linear_vels_x Linear_vels_y current speed of each vehicle on the track
    # collisions of each vehicle on the track
    # poses_x poses_y current position of each vehicle on the track
    # lap_counts number of laps of the circuit
    # lap_times time taken for a lap and current time of the lap
    # We only have 1 vehicle so we get the 0 for the speed of the singular vehicle
    # transform the reward based on the current speed of the vehicle
    # (TO-DO) Add incentive to go forward
    reward = reward*velVector(state['linear_vels_x'][0],state['linear_vels_y'][0]) if reward > 0 else reward
    # reduce reward if a collision happens
    reward -= 50 if state['collisions'].any() == 1.0 else 0
    return reward

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = 'f110-v0'

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = max_ep_len * 4      # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0008       # learning rate for actor network
    lr_critic = 0.003       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

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
    state_dim = preprocess_lidar(obs['scans'][0]).size + 1
    #print(state_dim)

    # action space dimension
    if has_continuous_action_space:
        action_dim = 2 # steer angle , velocity
    else:
        action_dim = env.action_space.n

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= max_training_timesteps:
        state, *_ = env.reset(poses=poses)
        current_ep_reward = 0

        for t in range(1, max_ep_len+1):

            actions = []
            processed_state = np.append(preprocess_lidar(state['scans'][0]),[velVector(state['linear_vels_x'][0],state['linear_vels_y'][0])])

            # select action with policy
            action = ppo_agent.select_action(processed_state)
            action[0] = max(min(action[0], 2), -2) # clamp between -2 to 2
            action[1] = abs(action[1])*10 # Max speed is around double the multiplier
            # steer angle , velocity
            actions.append(list(action))
            actions = np.array(actions)
            state, reward, done, _ = env.step(actions)
            reward = reward_fn(state,reward)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            if render:
                proc_ranges = state['scans'][0] #preprocess_lidar(state['scans'][0])
                vis.step(proc_ranges)
                env.render(mode='human')
                #time.sleep(frame_delay)
            
            # break; if the episode is over
            if done or state['collisions'].any() == 1.0 :
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()