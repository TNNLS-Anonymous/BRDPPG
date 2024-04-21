import torch
import numpy as np
import gym
from gym.spaces import Discrete
from policy import MlpPolicy, DiagonalGaussianMlpPolicy
from utils import get_inner_model, save_frames_as_gif
from utils import env_wrapper
import random


class Worker:

    def __init__(self,
                 id,
                 is_Byzantine,
                 env_name,
                 hidden_units,
                 gamma,
                 activation = 'Tanh',
                 output_activation = 'Identity',
                 attack_type = None,
                 max_epi_len = 0,
                 opts = None
                 ):
        super(Worker, self).__init__()
        
        # setup
        self.id = id
        self.is_Byzantine = is_Byzantine
        self.gamma = gamma
        # make environment, check spaces, get obs / act dims
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(0)
        self.attack_type = attack_type
        self.max_epi_len = max_epi_len
        self.opts = opts
        
        assert opts is not None
        
        # get observation dim
        obs_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, Discrete):
            n_acts = self.env.action_space.n
        else:
            n_acts = self.env.action_space.shape[0]
        
        hidden_sizes = list(eval(hidden_units))
        self.sizes = [obs_dim]+hidden_sizes+[n_acts] # make core of policy network
        
        # get policy net
        if isinstance(self.env.action_space, Discrete):
            self.logits_net = MlpPolicy(self.sizes, activation, output_activation)
        else:
            self.logits_net = DiagonalGaussianMlpPolicy(self.sizes, activation,)
        
        # optimizer
        # self.optimizer = torch.optim.Adam(self.logits_net.parameters(), lr = opts.lr_model * opts.num_worker)
        # self.optimizer = torch.optim.Adam(self.logits_net.parameters(), lr = opts.lr_model)
        self.scheduler = None
        if opts.optim == 'sgd':
            self.optimizer = torch.optim.SGD(self.logits_net.parameters(), lr=opts.lr_model, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(self.logits_net.parameters(), lr = opts.lr_model)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=320, gamma=0.1)
        
    
    def load_state_dict(self, param):
        self.logits_net.load_state_dict(param)

    def rollout(self, device, max_steps = 1000, render = False, env = None, obs = None, sample = True, mode = 'human', save_dir = './', filename = '.'):
        
        if env is None and obs is None:
            env = self.env
            obs = env.reset()
            
        done = False  
        ep_rew = []
        frames = []
        step = 0
        while not done and step < max_steps:
            step += 1
            if render:
                if mode == 'rgb':
                    frames.append(env.render(mode="rgb_array"))
                else:
                    env.render()
                
            # obs = env_wrapper(env.unwrapped.spec.id, obs)
            obs = env_wrapper(self.env_name, obs)
            action = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)[0]
            obs, rew, done, _ = env.step(action)
            ep_rew.append(rew)

        if mode == 'rgb': save_frames_as_gif(frames, save_dir, filename)
        return np.sum(ep_rew), len(ep_rew), ep_rew
    
    def collect_experience_for_training(self, B, device, record = False, sample = True, attack_type = None):
        # make some empty lists for logging.
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths
        batch_log_prob = []     # for gradient computing

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep
        
        # make two lists for recording the trajectory
        if record:
            batch_states = []
            batch_actions = []

        t = 1
        # collect experience by acting in the environment with current policy
        while True:
            # save trajectory
            if record:
                batch_states.append(obs)
            # act in the environment  
            obs = env_wrapper(self.env_name, obs)
            
            # simulate random-action attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'random_action':
                act_rnd = self.env.action_space.sample()
                if isinstance(act_rnd, int): # discrete action space
                    act_rnd = 0
                else: # continuous
                    act_rnd = np.zeros(len(self.env.action_space.sample()), dtype=np.float32) 
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample, fixed_action = act_rnd)
            else:
                act, log_prob = self.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
           
            obs, rew, done, info = self.env.step(act)
            
            # simulate reward-flipping attacker if needed
            if self.is_Byzantine and attack_type is not None and self.attack_type == 'reward-flipping': 
                rew = - rew
                
            # timestep
            t = t + 1
            
            # save action_log_prob, reward
            batch_log_prob.append(log_prob)
            
            ep_rews.append(rew)
            
            # save trajectory
            if record:
                batch_actions.append(act)

            if done or len(ep_rews) >= self.max_epi_len:
                
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                
                # the weight for each logprob(a_t|s_T) is sum_t^T (gamma^(t'-t) * r_t')
                returns = []
                R = 0
                # simulate random-reware attacker if needed
                if self.is_Byzantine and attack_type is not None and self.attack_type == 'random-reward': 
                    random.shuffle(ep_rews)
                    for r in ep_rews:
                        R = r + self.gamma * R
                        returns.insert(0, R)
                else:
                    for r in ep_rews[::-1]:
                        R = r + self.gamma * R
                        returns.insert(0, R)            
                returns = torch.tensor(returns, dtype=torch.float32)
                
                # return whitening
                advantage = (returns - returns.mean()) / (returns.std() + 1e-20)
                batch_weights += advantage

                # end experience loop if we have enough of it
                if len(batch_lens) >= B:
                    break
                
                # reset episode-specific variables
                obs, done, ep_rews, t = self.env.reset(), False, [], 1


        # make torch tensor and restrict to batch_size
        weights = torch.as_tensor(batch_weights, dtype = torch.float32).to(device)
        logp = torch.stack(batch_log_prob)

        if record:
            return weights, logp, batch_rets, batch_lens, batch_states, batch_actions
        else:
            return weights, logp, batch_rets, batch_lens
    
    
    def train_one_epoch(self, B, device, sample):
        
        # collect experience by acting in the environment with current policy
        weights, logp, batch_rets, batch_lens, batch_states, batch_actions = self.collect_experience_for_training(B, device, sample = sample, attack_type = self.attack_type, record=True)
        
        # self.weights = weights
        # self.logp = logp
        # self.batch_states = batch_states
        # self.batch_actions = batch_actions

        # calculate policy gradient loss
        batch_loss = -(logp * weights).mean()
    
        # take a single policy gradient update step
        self.logits_net.zero_grad()
        batch_loss.backward()
        # if self.opts.model_fusion:
        #     self.optimizer.step()
        
        # # determine if the agent is byzantine
        # if self.is_Byzantine and self.attack_type is not None:
        #     # return wrong gradient with noise
        #     param = []
        #     for item in self.parameters():
        #         if self.attack_type == 'zero-gradient':
        #             param.append(item.param * 0)
                
        #         elif self.attack_type == 'random-noise':
        #             rnd = (torch.rand(item.grad.shape, device = item.device) * 2 - 1) * (item.grad.max().data - item.grad.min().data) * 3
        #             param.append(item.grad + rnd)
                
        #         elif self.attack_type == 'sign-flipping':
        #             param.append(-2.5 * item.grad)
                    
        #         elif self.attack_type == 'reward-flipping':
        #             param.append(item.grad)
        #             # refer to collect_experience_for_training() to see attack

        #         elif self.attack_type == 'random-action':
        #             param.append(item.grad)
        #             # refer to collect_experience_for_training() to see attack
                
        #         elif self.attack_type == 'random-reward':
        #             param.append(item.grad)
        #             # refer to collect_experience_for_training() to see attack
                
        #         elif self.attack_type == 'FedScsPG-attack':
        #             param.append(item.grad)
        #             # refer to agent.py to see attack
                    
        #         else:
        #             pass

    
        # else:
            # return true gradient
        grad = [item.grad for item in self.parameters()]
        

        # report the results to the agent for training purpose
        return grad, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens), \
            batch_states, batch_actions, weights.clone().detach().numpy(), logp.clone().detach().numpy()

    def imp_samp(self, other_worker, device, sample=True):
        # Importance sampling
        other_logp = []
        for obs in self.batch_states:
            act, log_prob = other_worker.logits_net(torch.as_tensor(obs, dtype=torch.float32).to(device), sample = sample)
            other_logp.append(log_prob)

        ratios = []
        for other, current in zip(other_logp, self.batch_logp):
            ratio = np.exp(other.item() - current)
            ratios.append(ratio)

        imp_samp_weight = []
        for weight, ratio in zip(self.batch_weight, ratios):
            imp_samp_weight = weight * ratio
        
        return np.mean(imp_samp_weight), np.mean(self.batch_weight)

    def to(self, device):
        self.logits_net.to(device)
        return self
    
    def eval(self):
        self.logits_net.eval()
        return self
        
    def train(self):
        self.logits_net.train()
        return self
    
    def parameters(self):
        return self.logits_net.parameters()

    def state_dict(self):
        return self.logits_net.state_dict()

    def named_parameters(self):
        return self.logits_net.named_parameters()
