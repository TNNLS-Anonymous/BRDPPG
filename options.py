#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
import torch

def get_options(args=None):
    parser = argparse.ArgumentParser('FT-FedScsPG')

    ### Overall run settings
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1', choices = ['HalfCheetah-v2', 'LunarLander-v2', 'CartPole-v1'], 
                        help='OpenAI Gym env name for test')
    parser.add_argument('--eval_only', action='store_true', 
                        help='used only if to evaluate a pre-trained model')
    parser.add_argument('--no_saving', action='store_true', 
                        help='Disable saving checkpoints')
    parser.add_argument('--no_tb', action='store_true', 
                        help='Disable Tensorboard logging')
    parser.add_argument('--render', action='store_true', 
                        help='render to view the env')
    parser.add_argument('--mode', type=str, choices = ['human', 'rgb'], default='human', 
                        help='render mode')
    parser.add_argument('--log_dir', default = 'logs', 
                        help='log folder' )
    parser.add_argument('--run_name', default='run_name', 
                        help='Name to identify the experiments')
    
    
    # Multiple runs
    parser.add_argument('--multiple_run', type=int, default=1, 
                        help='number of repeated runs')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Starting point of random seed when running multiple times')

    
    # Federation and Byzantine parameters
    parser.add_argument('--num_worker', type=int, default=1, 
                        help = 'number of worker node')
    parser.add_argument('--num_Byzantine', type=int, default=0, 
                        help = 'number of worker node that is Byzantine')
    parser.add_argument('--alpha', type=float, default=0.4, 
                        help = 'atmost alpha-fractional worker nodes are Byzantine')
    parser.add_argument('--attack_type', type=str, default='non-attack', 
                        choices = ['zero-gradient', 'random-action', 'sign-flipping', 'reward-flipping', 'random-reward', 'random-noise', 'FedScsPG-attack','filtering-attack'],
                        help = 'the behavior scheme of a Byzantine worker')
    parser.add_argument('--conn_ratio', type=float, default=1., help='connectivity of graph')
    parser.add_argument('--batch', type=int, default=64., help='Batch size')
    parser.add_argument('--optim', type=str, default="sgd", help='Optimizer')
    
    # RL Algorithms (default GOMDP)
    parser.add_argument('--alg', type=str, default='GPOMDP',choices=['GPOMDP', 'SVRPG', 'FedPG_BR'], 
                        help='the algorithm')

    
    # Training and validating
    parser.add_argument('--val_size', type=int, default=10, 
                        help='Number of episoid used for reporting validation performance')
    parser.add_argument('--val_max_steps', type=int, default=1000, 
                        help='Maximum trajectory length used for reporting validation performance')
    

    # Load pre-trained modelss
    parser.add_argument('--load_path', default = None,
                        help='Path to load pre-trained model parameters')

    parser.add_argument('--fusion', type=str, default='model', choices = ['model', 'grad', 'grad_model'], 
                        help='Model fusion')

    parser.add_argument('--agg', type=str, default='weight', choices = ['weight', 'geomed', 'med', 'trimmed', 'ios'], 
                        help='Agggregation rules')
    parser.add_argument('--agg_obj', type=str, default='model', choices = ['model', 'grad', 'both'], 
                        help='Agggregation rules')
    parser.add_argument('--attack', type=str, default='none', choices = ['none', 'gaussian', 'sign_flip', 'zero', 'random_action'], 
                        help='attack modes')
    parser.add_argument('--qcut', type=int, default=0, help='number of cut in trimmed-mean')


    ### end of parameters
    opts = parser.parse_args(args)

    opts.use_cuda = False
    # opts.run_name = "{}_{}".format(opts.run_name, time.strftime("%Y%m%dT%H%M%S"))
    # opts.save_dir = os.path.join(
    #     'outputs',
    #     '{}'.format(opts.env_name),
    #     "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
    #     opts.alg,
    #     opts.run_name
    # ) if not opts.no_saving else None
    # opts.log_dir = os.path.join(
    #     f'{opts.log_dir}',
    #     '{}'.format(opts.env_name),
    #     "worker{}_byzantine{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.attack_type),
    #     opts.alg,
    #     opts.run_name
    # ) if not opts.no_tb else None
    
    if opts.env_name == 'CartPole-v1':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500
        opts.gamma  = 0.9
        opts.min_reward = 0  # for logging purpose (not important)
        opts.max_reward = 600  # for logging purpose (not important)

        # shared parameters
        opts.do_sample_for_training = True
        if opts.optim == 'sgd':
            opts.lr_model = 1e-1
        else:
            opts.lr_model = 1e-3
        opts.hidden_units = '16,16'
        opts.activation = 'ReLU'
        opts.output_activation = 'Tanh'
        
        # batch_size
        opts.B = opts.batch # for SVRPG and GOMDP
        opts.max_trajectories = opts.B * 300


  
    elif opts.env_name == 'HalfCheetah-v2':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 500  
        opts.max_trajectories = 1e4
        opts.gamma  = 0.995
        opts.min_reward = 0 # for logging purpose (not important)
        opts.max_reward = 4000 # for logging purpose (not important)
        
        # shared parameters
        opts.do_sample_for_training = True
        if opts.optim == 'sgd':
            opts.lr_model = 1e-1
        else:
            opts.lr_model = 1e-3
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'
       
        # batch_size
        opts.B = 96 # for SVRPG and GOMDP
        opts.max_trajectories = opts.B * 500

    if opts.env_name == 'LunarLander-v2':
        # Task-Specified Hyperparameters
        opts.max_epi_len = 1000  
        opts.max_trajectories = 1e4
        opts.gamma  = 0.99
        opts.min_reward = -1000 # for logging purpose (not important)
        opts.max_reward = 300 # for logging purpose (not important)
        
        # shared parameters
        opts.do_sample_for_training = True
        opts.lr_model = 1e-3 # 8e-4
        opts.hidden_units = '64,64'
        opts.activation = 'Tanh'
        opts.output_activation = 'Tanh'
        
        # batch_size
        opts.B = 32 # for SVRPG and GOMDP
        opts.max_trajectories = opts.B * 1000

    print(f'run {opts.alg}')
    
    return opts
