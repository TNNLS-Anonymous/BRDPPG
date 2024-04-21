from dis import dis
import enum
from re import A
import time
from multiprocessing import pool
import os
from tkinter import W
import numpy as np
import torch
import torch.optim as optim
from torch.multiprocessing import Pool
from tqdm import tqdm
from sklearn import metrics, neighbors
import matplotlib.pyplot as plt
from itertools import repeat
from scipy.interpolate import Rbf
import scipy.stats as st


from dec_worker import Worker
from utils import torch_load_cpu, get_inner_model, env_wrapper, gen_graph, metropolis_weight, geometric_median, reverse_flat, trimmed,IOS

class Data:
    def __init__(self) -> None:
        self.steps = {}
        self.eval_values = {}
        self.training_values = {}


def worker_run(worker, opts, batch_size, seed):
    worker.env.seed(seed)
    output = worker.train_one_epoch(batch_size, opts.device, opts.do_sample_for_training)

    return output


def worker_validate(worker, opts, w_id = 0, max_steps = 1000, render = False, run_id = 0, mode = 'humman'):
    val_ret = .0
    val_len = .0
    for _ in range(opts.val_size):
        epi_ret, epi_len, _ = worker.rollout(opts.device, 
                                             max_steps = max_steps, 
                                             render = render,
                                             sample=False,
                                             mode = mode,
                                             save_dir = './outputs/',
                                             filename = f'gym_{run_id}_{_}.gif')
        val_ret += epi_ret
        val_len += epi_len
    
    val_ret /= opts.val_size
    val_len /= opts.val_size
    
    return val_ret, val_len

def grad_assign(worker, grad, attack="none", device='cpu'):
    for i, param in enumerate(worker.parameters()):
        if attack == "none":
            param.grad = grad[i]
        elif attack == 'gaussian':
            param.grad = torch.normal(0, 100, size=grad[i].shape).to(device)
        elif attack == "sign_flip":
            param.grad = -2 * grad[i]
        elif attack == 'zero':
            param.grad = torch.zeros_like(grad[i])
        else:
            param.grad = grad[i]

def cal_param_difference(workers):
    param_matrix = [[] for _ in range(len(workers))]
    for i, w in enumerate(workers):
        for para in w.parameters():
            param_matrix[i].append(para.data.view(-1))
        param_matrix[i] = torch.cat(param_matrix[i])
    
    param_matrix = torch.stack(param_matrix)
    diff_matrix = param_matrix - torch.mean(param_matrix, dim=0)
    return torch.sum(diff_matrix)

def model_fusion(workers, comm_weight = None, agg='weight', Byzantine_workers=None, attack="none", device='cpu', qcut=0, ios=None):
    sign_flip_ratio=-2
    if agg == 'weight':
        param_dicts = [{} for _ in range(len(workers))]
        for key in workers[0].state_dict():
            param_matrix = []
            for w_id, w in enumerate(workers):
                if attack == 'None':
                    param_matrix.append(w.state_dict()[key])
                elif Byzantine_workers is not None and w_id in Byzantine_workers:
                    if attack == 'gaussian':
                        param_matrix.append(torch.normal(0, 100, size=w.state_dict()[key].shape).to(device))
                    elif attack == 'sign_flip':
                        param_matrix.append(w.state_dict()[key] *sign_flip_ratio)
                    elif attack == 'zero':
                        param_matrix.append(torch.zeros_like(w.state_dict()[key]))
                    else:
                        param_matrix.append(w.state_dict()[key])
                else:
                    param_matrix.append(w.state_dict()[key])
                
            param_matrix = torch.stack(param_matrix)
            if comm_weight is not None:
                ori_shape = param_matrix.shape
                new_param_matrix = torch.matmul(comm_weight, param_matrix.view(len(workers),-1)).reshape(ori_shape)
                for w_id, w in enumerate(workers):
                    param_dicts[w_id][key] = new_param_matrix[w_id]
            else:
                for w_id, w in enumerate(workers):
                    param_dicts[w_id][key] = torch.mean(param_matrix, dim=0)
        for w_id, w in enumerate(workers):
            w.load_state_dict(param_dicts[w_id])
    elif agg == 'geomed':
        params = {}
        for key in workers[0].state_dict():
            param_matrix = []
            for w_id, w in enumerate(workers):
                if attack == 'None':
                    param_matrix.append(w.state_dict()[key].view(-1))
                elif Byzantine_workers is not None and w_id in Byzantine_workers:
                    if attack == 'gaussian':
                        shape = w.state_dict()[key].view(-1).shape
                        param_matrix.append(torch.normal(0, 100, size=shape).to(device))
                    elif attack == 'sign_flip':
                        param_matrix.append(w.state_dict()[key].view(-1)*-1)
                    elif attack == 'zero':
                        param_matrix.append(torch.zeros_like(w.state_dict()[key]).view(-1))
                    else:
                        param_matrix.append(w.state_dict()[key].view(-1))
                else:
                    param_matrix.append(w.state_dict()[key].view(-1))
            param_matrix = torch.stack(param_matrix)
            params[key] = param_matrix
        
        adjac = (comm_weight > .0)
        agg_param_list = []
        for w_id, w in enumerate(workers):
            neigh = adjac[w_id]
            param_neigh = []
            for key in params:
                param_neigh.append(params[key][neigh])
            param_neigh = torch.cat(param_neigh, dim=1)
            flat_agg_param = geometric_median(param_neigh)
            agg_param = reverse_flat(workers[0].state_dict(), flat_agg_param)
            agg_param_list.append(agg_param)
        
        for w_id, w in enumerate(workers):
            w.load_state_dict(agg_param_list[w_id])    
    elif agg == 'med':
        params = {}
        for key in workers[0].state_dict():
            param_matrix = []
            for w_id, w in enumerate(workers):
                if attack == 'none':
                    param_matrix.append(w.state_dict()[key].view(-1))
                elif w_id in Byzantine_workers:
                    if attack == 'gaussian':
                        shape = w.state_dict()[key].view(-1).shape
                        param_matrix.append(torch.normal(0, 100, size=shape).to(device))
                    elif attack == 'sign_flip':
                        param_matrix.append(w.state_dict()[key].view(-1)*-1)
                    elif attack == 'zero':
                        param_matrix.append(torch.zeros_like(w.state_dict()[key]).view(-1))
                    else:
                        param_matrix.append(w.state_dict()[key].view(-1))
                else:
                    param_matrix.append(w.state_dict()[key].view(-1))
            param_matrix = torch.stack(param_matrix)
            params[key] = param_matrix
        
        adjac = (comm_weight > .0)
        agg_param_list = []
        for w_id, w in enumerate(workers):
            neigh = adjac[w_id]
            param_neigh = []
            for key in params:
                param_neigh.append(params[key][neigh])
            param_neigh = torch.cat(param_neigh, dim=1)
            flat_agg_param = torch.median(param_neigh, dim=0)[0]
            agg_param = reverse_flat(workers[0].state_dict(), flat_agg_param)
            agg_param_list.append(agg_param)
        for w_id, w in enumerate(workers):
            w.load_state_dict(agg_param_list[w_id])
    elif agg == 'trimmed':
        params = {}
        for key in workers[0].state_dict():
            param_matrix = []
            for w_id, w in enumerate(workers):
                if attack == 'none':
                    param_matrix.append(w.state_dict()[key].view(-1))
                elif w_id in Byzantine_workers:
                    if attack == 'gaussian':
                        shape = w.state_dict()[key].view(-1).shape
                        param_matrix.append(torch.normal(0, 1, size=shape).to(device))
                    elif attack == 'sign_flip':
                        param_matrix.append(w.state_dict()[key].view(-1)*sign_flip_ratio)
                    elif attack == 'zero':
                        param_matrix.append(torch.zeros_like(w.state_dict()[key]).view(-1))
                    else:
                        param_matrix.append(w.state_dict()[key].view(-1))
                else:
                    param_matrix.append(w.state_dict()[key].view(-1))
            param_matrix = torch.stack(param_matrix)
            params[key] = param_matrix
        
        adjac = (comm_weight > .0)
        agg_param_list = []
        for w_id, w in enumerate(workers):
            neigh = adjac[w_id]
            neigh[w_id] = False
            param_neigh = []
            self_param = []
            for key in params:
                param_neigh.append(params[key][neigh])
                self_param.append(params[key][w_id])
            param_neigh = torch.cat(param_neigh, dim=1)
            self_param = torch.cat(self_param).view(1,-1)
            trimmed_param = None
            if attack == 'sign_flip':
                if w_id == 7:
                    trimmed_param = trimmed(param_neigh, 1)
                else:
                    trimmed_param = trimmed(param_neigh, 1)
            elif attack == 'gaussian':
                trimmed_param = trimmed(param_neigh, 1)
            else:
                trimmed_param = trimmed(param_neigh, 1)
            total_trimmed_param = torch.cat((trimmed_param, self_param), dim=0)
            flat_agg_param = torch.mean(total_trimmed_param, dim=0)
            agg_param = reverse_flat(workers[0].state_dict(), flat_agg_param)
            agg_param_list.append(agg_param)
        for w_id, w in enumerate(workers):
            w.load_state_dict(agg_param_list[w_id])    
    
    elif agg == 'ios':
        params = {}
        for key in workers[0].state_dict():
            param_matrix = []
            for w_id, w in enumerate(workers):
                if attack == 'none':
                    param_matrix.append(w.state_dict()[key].view(-1))
                elif w_id in Byzantine_workers:
                    if attack == 'gaussian':
                        shape = w.state_dict()[key].view(-1).shape
                        param_matrix.append(torch.normal(0, 100, size=shape).to(device))
                    elif attack == 'sign_flip':
                        param_matrix.append(w.state_dict()[key].view(-1)*-1)
                    elif attack == 'zero':
                        param_matrix.append(torch.zeros_like(w.state_dict()[key]).view(-1))
                    else:
                        param_matrix.append(w.state_dict()[key].view(-1))
                else:
                    param_matrix.append(w.state_dict()[key].view(-1))
            param_matrix = torch.stack(param_matrix)
            params[key] = param_matrix
        
        # adjac = (comm_weight > .0)
        # agg_param_list = []
        # for w_id, w in enumerate(workers):
        #     neigh = adjac[w_id]
        #     param_neigh = []
        #     for key in params:
        #         param_neigh.append(params[key][neigh])
        #     param_neigh = torch.cat(param_neigh, dim=1)
        #     flat_agg_param = trimmed_mean(param_neigh, qcut)
        #     agg_param = reverse_flat(workers[0].state_dict(), flat_agg_param)
        #     agg_param_list.append(agg_param)
        
        total_params = []
        for key in params:
            total_params.append(params[key])
        total_params = torch.cat(total_params, dim=1)
        assert ios is not None
        agg_param_list = []
        for w_id, w in enumerate(workers):
            flat_agg_param = ios.run(total_params, w_id)
            agg_param = reverse_flat(workers[0].state_dict(), flat_agg_param)
            agg_param_list.append(agg_param)

        for w_id, w in enumerate(workers):
            w.load_state_dict(agg_param_list[w_id])
    

def average_model(workers, virtual_worker):
    virtual_param_dict = {}
    for key in workers[0].state_dict():
        param_matrix = []
        for w in workers:
            param_matrix.append(w.state_dict()[key])
        param_matrix = torch.stack(param_matrix)
        virtual_param_dict[key] = torch.mean(param_matrix, dim=0)
    
    if virtual_worker is not None:
        virtual_worker.load_state_dict(virtual_param_dict)


def grad_fusion(workers, grads, comm_weight = None, agg='weight'):
    if agg == 'weight':
        agg_grad = []
        for idx, item in enumerate(workers[0].parameters()):
            grad_item = []
            for i in range(len(workers)):
                grad_item.append(grads[i][idx])
            
            if comm_weight is not None:
                grad = torch.stack(grad_item)
                ori_shape = grad.shape
                grad = grad.view(ori_shape[0], -1)
                agg_grad.append(torch.matmul(comm_weight, grad).view(ori_shape))
            else:
                agg_grad.append(torch.stack(grad_item).mean(0))
        
        for w_idx, w in enumerate(workers):
            for idx, param in enumerate(w.parameters()):
                if comm_weight is not None:
                    param.grad = agg_grad[idx][w_idx]
                else:
                    param.grad = agg_grad[idx]
    elif agg == 'geomed':
        grads = {}
        for w in workers:
            for name, para in w.named_parameters():
                if name not in grads:
                    grads[name] = [para.grad.view(-1)]
                else:
                    grads[name].append(para.grad.view(-1))
        for name in grads:
            grads[name] = torch.stack(grads[name])
        
        adjac = (comm_weight > .0)
        agg_list = []
        for w_id, w in enumerate(workers):
            neigh = adjac[w_id]
            grad_neigh = []
            for name in grads:
                grad_neigh.append(grads[name][neigh])
            grad_neigh = torch.cat(grad_neigh, dim=1)
            flat_agg_grad = geometric_median(grad_neigh)
            agg_grad = reverse_flat(workers[0].state_dict(), flat_agg_grad)
            agg_list.append(agg_grad)
        
        for w_id, w in enumerate(workers):
            for name, para in w.named_parameters():
                para.grad = agg_list[w_id][name]
    elif agg == 'med':
        grads = {}
        for w in workers:
            for name, para in w.named_parameters():
                if name not in grads:
                    grads[name] = [para.grad.view(-1)]
                else:
                    grads[name].append(para.grad.view(-1))
        for name in grads:
            grads[name] = torch.stack(grads[name])
        
        adjac = (comm_weight > .0)
        agg_list = []
        for w_id, w in enumerate(workers):
            neigh = adjac[w_id]
            grad_neigh = []
            for name in grads:
                grad_neigh.append(grads[name][neigh])
            grad_neigh = torch.cat(grad_neigh, dim=1)
            flat_agg_grad = torch.median(grad_neigh, dim=0)[0]
            agg_grad = reverse_flat(workers[0].state_dict(), flat_agg_grad)
            agg_list.append(agg_grad)
        
        for w_id, w in enumerate(workers):
            for name, para in w.named_parameters():
                para.grad = agg_list[w_id][name]

    
    for w in workers:
        w.optimizer.step()
        if w.scheduler is not None:
            w.scheduler.step()



def grad_model_fusion(workers, grads, comm_weight=None, agg='weight', agg_obj='model', Byzantine_workers=None, attack='none', device='cpu'):
    grad_fusion(workers, grads, comm_weight, 'weight')
    model_fusion(workers, comm_weight, agg, Byzantine_workers, attack, device)
    # if agg_obj == 'model':
    #     model_fusion(workers, comm_weight, agg, Byzantine_workers, attack, device)
    #     grad_fusion(workers, grads, comm_weight, 'weight')
    # elif agg_obj == 'grad':
    #     model_fusion(workers, comm_weight, 'weight', Byzantine_workers, attack, device)
    #     grad_fusion(workers, grads, comm_weight, agg)
    # elif agg_obj =='both':
    #     model_fusion(workers, comm_weight, agg, Byzantine_workers, attack, device)
    #     grad_fusion(workers, grads, comm_weight, agg)


def cal_dist(workers, comm_weight, tb_logger=None, run_id=0, step=0):
    adj_matrix = (comm_weight > .0).float()
    param_dicts = [{} for _ in range(len(workers))]
    dist = [[] for _ in range(len(workers))]
    for key in workers[0].state_dict():
        param_matrix = []
        for w in workers:
            param_matrix.append(w.state_dict()[key])
        param_matrix = torch.stack(param_matrix).view(len(workers), -1)
        new_param_matrix = torch.matmul(adj_matrix, param_matrix.view(len(workers), -1))/ torch.sum(adj_matrix, dim=1, keepdim=True)
        for w_id, w in enumerate(workers):
            dist[w_id].append((param_matrix[w_id]-new_param_matrix[w_id]))
    
    diff = []
    for d in dist:
        diff_param = torch.cat(d)
        diff.append(torch.norm(diff_param))
    
    for w_id, v in enumerate(diff):
        tb_logger.add_scalar(f'train/param_diff_worker_{w_id}_run_id_{run_id}', v, step)
        

def imp_samp(workers, device, tb_logger = None, step=0, run_id=0):
    # Importance sampling
    for w_id, w in enumerate(workers):
        for _w_id, _w in enumerate(workers):
            o_w, c_w = w.imp_samp(_w, device)
            if tb_logger is not None:
                tb_logger.add_scalar(f"Woker_{w_id}_training_run_id_{run_id}/{w_id}_to{_w_id}", o_w-c_w, step)
    


    
                                    

class DecSimulation(object):
    def __init__(self, opts, graph=None) -> None:
        
        self.opts = opts
        self.num_worker = opts.num_worker
        self.num_Byzantine = opts.num_Byzantine

        if graph is not None:
            self.graph = graph

        # TODO: training
        self.pool = Pool(self.num_worker)
        self.data = Data()
    
    def reset(self):
        # self.graph = gen_graph(self.num_worker, range(self.opts.num_Byzantine), self.opts.log_dir, self.opts.conn_ratio)
        self.weight_matrix = metropolis_weight(self.graph)
        self.weight_matrix = torch.tensor(self.weight_matrix, dtype=torch.float32).to(self.opts.device)
        adjac_graph = (self.weight_matrix > .0)
        self.ios = IOS(adjac_graph, self.num_worker, self.num_Byzantine)

        self.workers = []
        
        self.virtual_worker = None
    
        self.Byzantine_workers = np.random.choice(list(range(self.num_worker)), self.num_Byzantine, replace=False)
        self.Byzantine_indicator = []
        for i in range(self.num_worker+1):
            self.Byzantine_indicator.append(True if i in self.Byzantine_workers else False)
            worker = Worker(
                id = i,
                is_Byzantine = self.Byzantine_indicator[i],
                env_name = self.opts.env_name,
                gamma = self.opts.gamma,
                hidden_units = self.opts.hidden_units,
                activation = self.opts.activation,
                output_activation = self.opts.output_activation,
                attack_type = self.opts.attack,
                max_epi_len = self.opts.max_epi_len,
                opts = self.opts
            ).to(self.opts.device)
            if i==self.num_worker:
                self.virtual_worker = worker
            else:
                self.workers.append(worker)
         
        for i in range(self.num_worker):
            self.workers[i].load_state_dict(self.virtual_worker.state_dict())
        print(f'{self.opts.num_worker} workers initilized with {self.opts.num_Byzantine if self.opts.num_Byzantine >0 else "None"} of them are Byzantine.')


    def load(self, load_path):
        # TODO
        pass

    def save(self, epoch, run_id):
        print('Saving model and state ...')
        pass

    def eval(self):
        for w in self.workers:
            w.eval()

    def train(self):
        for w in self.workers:
            w.train()

    def start_training(self, tb_logger = None, run_id = None):
        opts = self.opts

        # for storing number of trajectories sampled
        step = 0
        num_iter = 0
        ratios_step = 0
        start_time = time.time()



        # Start the training loop
        cal_dist(self.workers, self.weight_matrix, tb_logger, run_id, 0)
        while step <= opts.max_trajectories:
            # epoch for storing checkpoints of model
            num_iter += 1
            print('\n')
            print("|",format(f" Training epoch {num_iter} run_id {run_id} in {opts.seeds}","*^60"),"|")
            print('Total time elapsed: {:.1f} min {:.0f} s'.format((time.time()-start_time)//60, (time.time()-start_time)%60))
            # Turn model into training mode
            self.train()

            # some empty list for training and logging
            batch_loss = []
            batch_rets = []
            batch_lens = []
            batch_grad = []

            #
            batch_size = opts.B
            seeds = np.random.randint(1, 100000, self.num_worker).tolist()
            args = zip(self.workers, repeat(opts), repeat(batch_size), seeds)

            # multipe process to run 
            results = self.pool.starmap(worker_run, args)

            w_idx = 0
            for output in tqdm(results, desc='Worker node'):
                grad, loss, rets, lens, state, actions, weigths, logp = output
                
                if opts.attack == 'none':
                    grad_assign(self.workers[w_idx], grad)
                elif w_idx in self.Byzantine_workers:
                    grad_assign(self.workers[w_idx], grad, opts.attack, opts.device)
                else:
                    grad_assign(self.workers[w_idx], grad)
                
                self.workers[w_idx].batch_states = state
                self.workers[w_idx].batch_actions = actions
                self.workers[w_idx].batch_weight = weigths
                self.workers[w_idx].batch_logp = logp
                w_idx +=1
                # store all values
                batch_loss.append(loss)
                batch_rets.append(rets)
                batch_lens.append(lens)
                batch_grad.append(grad)

            # simulate decentralized communication


            # aggregate models
            if opts.fusion == 'model':
                cal_param_difference(self.workers)
                for w in self.workers:
                    w.optimizer.step()
                model_fusion(self.workers, self.weight_matrix, agg=opts.agg, qcut=opts.qcut, ios=self.ios, attack=opts.attack, Byzantine_workers=self.Byzantine_workers)
            elif opts.fusion == 'grad':
                grad_fusion(self.workers, batch_grad, self.weight_matrix)
            elif opts.fusion == 'grad_model':
                grad_model_fusion(self.workers, batch_grad, self.weight_matrix, agg=opts.agg, agg_obj=opts.agg_obj,
                                  Byzantine_workers=self.Byzantine_workers, attack=opts.attack, device=opts.device)

            
            step += batch_size

            # imp_samp(self.workers, opts.device, tb_logger, step, run_id)
            cal_dist(self.workers, self.weight_matrix, tb_logger, run_id, num_iter)

            # logging to tensorboard   
            if tb_logger is not None:
                # trainning log
                tb_logger.add_scalar(f'train/total_rewards_run_id_{run_id}', np.mean(batch_rets), num_iter)
                tb_logger.add_scalar(f'train/epi_length_run_id_{run_id}', np.mean(batch_lens), num_iter)
                tb_logger.add_scalar(f'train/loss_run_id_{run_id}', np.mean(batch_loss), num_iter)
                for w_id, w in enumerate(self.workers):
                    tb_logger.add_scalar(f'Woker_{w_id}_training__run_id_{run_id}/lr', w.optimizer.param_groups[0]['lr'], num_iter)
            

            if run_id not in self.data.steps.keys():
                self.data.steps[run_id] = []
                self.data.eval_values[run_id] = []
                self.data.training_values[run_id] = []
            self.data.steps[run_id].append(num_iter)
            self.data.training_values[run_id].append(np.mean(batch_rets))

            # do validating
            eval_return = self.validate(num_iter, tb_logger, run_id)
            if(tb_logger is not None):
                 self.data.eval_values[run_id].append(eval_return)
        
            if not opts.no_saving:
                self.save(step, run_id)



    def validate(self, epoch, tb_logger = None, run_id = 0):
        args = zip(self.workers,
                   repeat(self.opts),
                   range(self.num_worker),
                   repeat(self.opts.val_max_steps),
                   repeat(self.opts.render),    
                   repeat(run_id),
                   repeat(self.opts.mode)
                   )
        val_res = self.pool.starmap(worker_validate, args)
        if tb_logger is not None:
            worker_id = 0
            ret = .0
            for res in tqdm(val_res, desc='Validate Woker node'):
                val_ret, val_len = res  
                ret += val_ret
                tb_logger.add_scalar(f'validate/total_rewards_worker_{worker_id}_run_id_{run_id}', val_ret, epoch)
                tb_logger.add_scalar(f'validate/epi_length_worker_{worker_id}_run_id_{run_id}', val_len, epoch)
                worker_id += 1
            # tb_logger.close()
            ret /= self.num_worker
            print(f"Validate return: {ret}")
        return ret

        # average_model(self.workers, self.virtual_worker)
        # val_ret, val_len = worker_validate(self.virtual_worker, self.opts, 0, 
        #                                    self.opts.val_max_steps, self.opts.render, 
        #                                    run_id, self.opts.mode)
        # if tb_logger is not None:
        #     tb_logger.add_scalar(f'validate/total_rewards_run_id_{run_id}', val_ret, epoch)
        #     tb_logger.add_scalar(f'validate/epi_length_run_id_{run_id}', val_len, epoch)
        #     print(f"Validate return: {val_ret}")
        # return val_ret

    def plot_graph(self, array, save_dir=None, run_id=0):
        plt.ioff()
        fig = plt.figure(figsize=(8,4))
        y = []
        
        for _id in self.data.steps.keys():
             x = self.data.steps[_id]
             y.append(Rbf(x, array[_id], function = 'linear')(np.arange(self.opts.max_trajectories//self.opts.B)))
        

        mean = np.mean(y, axis=0)

        if save_dir  is not None:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            np_x = []
            np_y = []
            for _id in self.data.steps.keys():
                np_x.append(self.data.steps[_id])
                np_y.append(array[_id])
            np_x = np.asarray(np_x)
            np_y = np.asarray(np_y)
            np.save(os.path.join(save_dir, 'x_{}.npy'.format(run_id)), np_x)
            np.save(os.path.join(save_dir, 'y_{}.npy'.format(run_id)), np_y)
        
        l, h = st.norm.interval(0.90, loc=np.mean(y, axis = 0), scale=st.sem(y, axis = 0))
        
        plt.plot(mean)
        plt.fill_between(range(int(self.opts.max_trajectories//self.opts.B)), l, h, alpha = 0.5)

        plt.ylim([self.opts.min_reward, self.opts.max_reward])
        
        plt.xlabel("Number of Trajectories")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        return fig
        
    def log_performance(self, tb_logger=None, save_dir=None, run_id=0):
        eval_img = self.plot_graph(self.data.eval_values, save_dir=os.path.join(save_dir, "eval"), run_id=run_id)
        training_img = self.plot_graph(self.data.training_values, save_dir=os.path.join(save_dir, "train"), run_id=run_id)
        tb_logger.add_figure(f'validate/performance_until_{len(self.data.steps.keys())}_runs', eval_img, len(self.data.steps.keys()))
        tb_logger.add_figure(f'train/performance_until_{len(self.data.steps.keys())}_runs', training_img, len(self.data.steps.keys()))        