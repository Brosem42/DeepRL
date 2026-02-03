# import
import os
os.environ['OMP_NUM_THREADS']='1' # set number of CPU threads to 1
import torch
import argparse
from env import create_train_env
from model import ActorCritic
from optimizer import GlobalAdam
from process import local_train, local_test
import torch.multiprocessing as _mp
import shutil

# get args such as env name, level of game, action type, hyperparmaeters for RL, lr, and process related to parallelization
def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model: Asynchronous Methods for Deep Reinforcement Learning""""
    )
    parser.add_argument('--world', type=int, default=1)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--action-type', type=str, default='complex')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--num_local_steps', type=int, default=50)
    parser.add_argument('--num_global_steps', type=int, default=5e6)
    parser.add_argument('--num_processes', type=int, default=6)
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--max_actions', type=int, default=200, help='Maximum repetition steps in test phase')
    parser.add_argument('--log_path', type=str, default='tensorboard/A3CSuperMarioLogs')
    parser.add_argument('--saved_path', type=str, default='trained_models')
    parser.add_argument('--load_path_from_previous_stage', type=bool, default=False, help='Load weight from previous trained stage')

    parser.add_argument('--use_gpu', type=bool, default=False)
    args = parser.parse_args()
    return args


#training environment 
def train(opt):
    