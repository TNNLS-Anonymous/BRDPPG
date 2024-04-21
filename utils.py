import torch
import numpy as np
import networkx as nx

from torch.nn import DataParallel

from matplotlib import animation
import matplotlib.pyplot as plt

def torch_load_cpu(load_path):
    return torch.load(load_path, map_location=lambda storage, loc: storage)  # Load on CPU

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model

def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)

def env_wrapper(name, obs):
    return obs

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=120)

def gen_graph(nodeSize, byzantine, save_dir=None, conn_ratio=1.):
    """
    Randomly generate a graph where the regular workers are connected.

    :param nodeSize: the number of workers
    :param byzantine: the set of Byzantine workers
    """
    while True:
        G = nx.fast_gnp_random_graph(nodeSize, conn_ratio)
        H = G.copy()
        for i in byzantine:
            H.remove_node(i)
        num_connected = 0
        for _ in nx.connected_components(H):
            num_connected += 1
        if num_connected == 1:
            # nx.draw(G)
            # plt.show()
            break
    if save_dir is not None:
        import os
        plt.clf()
        pos = nx.spring_layout(G)
        nx.draw(G, pos=pos, font_weight='bold')
        plt.savefig(os.path.join(save_dir, 'graph.pdf'))
    return G

def metropolis_weight(G):
    num = len(G)
    matrix = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if (i, j) in G.edges() and i != j:
                matrix[i][j] = 1 / (1 + max(G.degree[i], G.degree[j]))

    for i in range(num):
        for j in range(num):
            if i == j:
                matrix[i][j] = 1 - sum(matrix[i])

    return matrix

def geometric_median(wList):
    max_iter = 80
    tol = 1e-5
    guess = torch.mean(wList, dim=0)
    for _ in range(max_iter):
        dist_li = torch.norm(wList-guess, dim=1)
        for i in range(len(dist_li)):
            if dist_li[i] == 0:
                dist_li[i] = 1
        temp1 = torch.sum(torch.stack([w/d for w, d in zip(wList, dist_li)]), dim=0)
        temp2 = torch.sum(1/dist_li)
        guess_next = temp1 / temp2
        guess_movement = torch.norm(guess - guess_next)
        guess = guess_next
        if guess_movement <= tol:
            break
    return guess

def trimmed(wList, q):
    sorted_w, _ = torch.sort(wList, dim=0)
    if q > 0 and 2 * q < sorted_w.shape[0]:
        trimmed_w = sorted_w[q:-q]
    elif q == 0:
        trimmed_w = sorted_w
    else:
        raise Exception('Too many discard')
    return trimmed_w

def reverse_flat(elem_dict, flat_elem):
    temp_flat_size = []
    for elem_name in elem_dict:
        ele_size = elem_dict[elem_name].nelement()
        temp_flat_size.append(ele_size)
    start = 0
    index = 0
    new_elem_dict = {}
    for elem_name in elem_dict:
        offset = temp_flat_size[index]
        new_elem  = torch.reshape(flat_elem[start:start+offset], shape=elem_dict[elem_name].shape)
        new_elem_dict[elem_name] = new_elem
        start += offset
        index += 1
    return new_elem_dict

def copy_flat_grads(params):
    new_grads = []

    for param in params:
        g = torch.clone(param.grad).detach().view(-1)
        new_grads.append(g)
    new_grads = torch.cat(new_grads)
    return new_grads

def copy_flat_param(params):
    new_grads = []

    for param in params:
        g = torch.clone(param.grad).detach().view(-1)
        new_grads.append(g)
    new_grads = torch.cat(new_grads)
    return new_grads

class IOS():
    def __init__(self, adjac_graph, node_size, byz_cnt=-1):
        self.W = torch.eye(node_size, dtype=torch.float)
        self.Byz_cnt = byz_cnt
        self.adjac_graph = adjac_graph

        for i in range(node_size):
            for j in range(node_size):
                if i == j or not adjac_graph[j,i]:
                    continue
                i_n = torch.sum(adjac_graph[i])
                j_n = torch.sum(adjac_graph[j])
                self.W[i][j] = 1 / max(i_n, j_n)
                # self.W[i][j] = 1 / i_n
                self.W[i][i] -= self.W[i][j]
    
    def run(self, local_models, node):
        remain_models = local_models[self.adjac_graph[node]]
        remain_weight = self.W[node][self.adjac_graph[node]]

        for _ in range(self.Byz_cnt):
            mean = torch.tensordot(remain_weight, remain_models, dims=1)
            mean /= remain_weight.sum()
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_models
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_models.size(0)) != remove_idx
            remain_models = remain_models[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_models, dims=1)
        res /= remain_weight.sum()

        
        return res