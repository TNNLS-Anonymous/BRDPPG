#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from operator import imod
import torch
import os
import json
import torch
import pprint
import numpy as np
import random
import time
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

from agent import Agent
from worker import Worker
from options import get_options
from utils import get_inner_model, gen_graph
from dec import DecSimulation

import multiprocessing as mp

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def run(opts):
    seed_torch()
    # Pretty print the run args
    pprint.pprint(vars(opts))
    
    
    
    # Figure out the RL
    
    
    # Do validation only
    if opts.eval_only:
        
        # setup tensorboard
        if not opts.no_tb:
            tb_writer = SummaryWriter(opts.log_dir)
        else:
            tb_writer = None
        dec_sim = DecSimulation(opts)

        # Set the random seed
        seed_torch(0)
        dec_sim.reset()
        
        # Load data from load_path
        if opts.load_path is not None:
            dec_sim.load(opts.load_path)
        
        dec_sim.start_validating(tb_writer, 0, opts.val_max_steps, opts.render, mode = opts.mode)
        
    else:
        # Optionally configure tensorboard
        # Configure for multiple runs    
        assert opts.multiple_run > 0
        seeds = (np.arange(opts.multiple_run) + opts.seed ).tolist()
        run_name = "run_{}".format(time.strftime("%Y%m%dT%H%M%S"))
        save_dir = os.path.join(
                'outputs',
                '{}'.format(opts.env_name),
                "worker{}_byzantine{}_conn_{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.conn_ratio, opts.attack),
                opts.alg,
                run_name
            ) if not opts.no_saving else None
        if not opts.no_saving and not os.path.exists(save_dir):
                os.makedirs(save_dir)
        # Save arguments so exact configuration can always be found
        if not opts.no_saving:
                with open(os.path.join(save_dir, "args.json"), 'w') as f:
                    json.dump(vars(opts), f, indent=True)
        log_dir = os.path.join(
                f'{opts.log_dir}',
                '{}'.format(opts.env_name),
                "worker{}_byzantine{}_conn_{}_{}".format(opts.num_worker, opts.num_Byzantine, opts.conn_ratio, opts.attack),
                opts.alg,
                run_name
            ) if not opts.no_tb else None
            # setup tensorboard
        if not opts.no_tb:
            tb_writer = SummaryWriter(log_dir)
        else:
            tb_writer = None
        new_opts = deepcopy(opts)
        new_opts.save_dir = save_dir
        new_opts.log_dir = log_dir
        # Set the device
        new_opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
        #new_opts.device = torch.device("cuda:0")
        new_opts.seeds = seeds

        #graph = gen_graph(new_opts.num_worker, (7,), new_opts.log_dir, new_opts.conn_ratio)
        graph = gen_graph(new_opts.num_worker, [4], new_opts.log_dir, new_opts.conn_ratio)
        dec_sim = DecSimulation(new_opts, graph)
        for run_id in seeds:
            # Set the random seed
            seed_torch(run_id)
            dec_sim.reset()

            # Start training here
            dec_sim.start_training(tb_writer, run_id)
            if tb_writer:
                dec_sim.log_performance(tb_writer, save_dir=log_dir, run_id=run_id)
            


if __name__ == "__main__":
    seed_torch()
    import warnings
    warnings.filterwarnings("ignore", category=Warning)
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    mp.set_start_method('spawn')
    run(get_options())
