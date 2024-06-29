from abc import ABC, abstractmethod
from melee import enums
import numpy as np
from melee_env.agents.util import *
import threading
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import numpy as np
from tqdm import tqdm
import random
from melee_env.agents.util import ObservationSpace, ActionSpace
import model
from torch.distributions.categorical import Categorical
from buffer import TrajBuffer

from hyper_param import PARAMS
class Agent(ABC):
    def __init__(self):
        self.agent_type = "AI"
        self.controller = None
        self.port = None  # this is also in controller, maybe redundant?
        self.action = 0
        self.press_start = False
        self.self_observation = None
        self.current_frame = 0

    @abstractmethod
    def act(self):
        pass
    
class OnlineDecisionTransformerAgent(Agent, nn.Module):
    def __init__(self, obs_space):
        super().__init__()
        self.s_dim = 6
        self.a_dim = 45
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.Decision_transformer(s_dim=self.s_dim, a_dim=self.a_dim, hidden_dim=64, K=self.K, max_timestep=self.max_timestep).to(self.device)
        self.register_buffer(name='lmbda', tensor=torch.tensor([1]), persistent=True)
        self.character = enums.Character.FOX
        self.action_space = ActionSpace()
        self.observation_space = obs_space

        #hyperparameters
        self.objective_R = 1
        self.max_timestep = 4096
        self.K = 128
        self.buffer_size = 32
        self.batch_size = 16
        self.training_iter = 8
        self.lmbda_lr = 0.001 
        self.beta = torch.tensor[1]

        #buffer and optimizer
        self.traj_buffer = TrajBuffer(K = self.K, buffer_size = self.buffer_size, a_dim=self.a_dim)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())    
           

        #reset after each episodes
        self.timestep = 0
        self.register_buffer(name='R', persistent=False) #temporal buffers
        self.register_buffer(name='s', persistent=False) #to contain R, s, a of current episode
        self.register_buffer(name='a', persistent=False)
    
    @from_action_space
    @from_observation_space
    def act(self, observation):
        obs, reward, done, info = observation
        self.push_R(reward)
        self.push_s(obs)
        R, s, a= self.get_Rsa_seq()
        t = (torch.arange(self.timestep-self.K+1, self.timestep) if self.timestep-self.K>0 else torch.arange(0, self.K)).unsqueeze(0).to(self.device)
        _, _, a_preds = self.model.forward(t=t, R=R, s=s, a=a)
        self.timestep += 1

        a_pred = Categorical(probs=a_preds[0, self.current_t]).sample()
        self.push_a(a_pred)
        return a_pred
        
    def train(self):
        for _ in range(self.training_iter):
            self.optimizer.zero_grad()
            T, R, s, a= self.traj_buffer.get_trajs(self.batch_size)
            _, _, a_preds = self.model.forward(T, R, s, a)
            lmbda = self.buffers()['lmbda']
            L, dlmbda = self.lagrangian(a_preds, a, lmbda=lmbda) #to ensure its shannon entropy is larger than beta, a fixed const,
            L.backward()                                         #we optimize Lagrangian associated with both loss fn and entropy.
            self.optimizer.step() #optimizing theta, parameters of the model
            lmbda = max(lmbda + self.lmbda_lr * dlmbda, 0) #optimizing lmbda

    def end_ep(self):
        self.update_traj_buffer()
        self.timestep = 0
        self.buffers()['R'] = None
        self.buffers()['s'] = None
        self.buffers()['a'] = None


    
    def loss_fn(self, a, a_preds):
        return nn.CrossEntropyLoss()(a, a_preds)

    def lagrangian(self, a_preds, a, lmbda):
        J = self.loss_fn(a, a_preds)
        H = torch.sum(Categorical(probs = a_preds).entropy() / self.K)
        H_ = self.beta - H
        return J + lmbda * H_, H_



    def update_traj_buffer(self):
        R = self.buffers()['R'].clone().detach().numpy()
        s = self.buffers()['s'].clone().detach().numpy()
        a = self.buffers()['a'].clone().detach()
        r = 0
        for i in range(len(s)).reversed():
            r += reward(s[i]) #self.observationspace.reward(state) 
            R[i] = r #Hindsight Return Relabeling
        R = torch.tensor(R)
        s = torch.tensor(s)
        self.traj_buffer.push_traj(R, s, a)
        


    def push_R(self, r):
        if self.buffers()['R'] is None:
            self.buffers()['R'] = torch.tensor([[self.objective_R]])
        else:
            RtG = self.buffers()['R'][-1].item() - r
            self.buffers()['R'] = torch.cat(self.buffer['R'], torch.tensor[[RtG]], dim=0)
    
    def push_s(self, s):
        if self.buffers()['s'] is None:
            self.buffers()['s'] = s.flatten(0,1).unsqueeze(0)
        else:
            self.buffers()['s'] = torch.cat(self.buffers['s'], s.flatten(0,1).unsqueeze(0), dim=0)

    def push_a(self, a):
        if self.buffers()['a'] is None:
            self.buffers()['a'] = a.unsqueeze(0)
        else:
            self.buffers()['a'] = torch.cat(self.buffers()['a'], a.unsqueeze(0), dim=0)

    def add_padding(self, seq):
        _, T, h = seq.shape
        if self.K - T > 0:
            seq = torch.cat(seq, torch.zeros(1, self.K - T, h), dim=1)
        return seq
    
    def get_Rsa_seq(self):
        i = max(self.timestep - self.K + 1, 0)
        t = self.add_padding(self.buffers()['t'][i:-1]).unsqueeze(0)
        R = self.add_padding(self.buffers()['R'][i:-1]).unsqueeze(0)
        s = self.add_padding(self.buffers()['s'][i:-1]).unsqueeze(0)
        if self.buffers()['a'] is not None:
            a = self.add_padding(F.one_hot(self.buffers()['a'][i - 1:-1], num_classes=self.a_dim)).unsqueeze(0)
        else:
            a = torch.zeros(1, self.K, self.a_dim).unsqueeze(0)
        return R, s, a