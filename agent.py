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
from model import Decision_transformer
from torch.distributions.categorical import Categorical
from buffer import TrajBuffer

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
    
class OnlineDecisionTransformerAgent(Agent):
    def __init__(self, obs_space):
        super().__init__() #super().__init__() doesn't work??
        #hyperparameters
        self.s_dim = 37
        self.a_dim = 28
        self.objective_R = 50
        self.max_timestep = 32768
        self.K = 128
        self.buffer_size = 64
        self.batch_size = 16
        self.training_iter = 4
        self.lmbda_lr = 0.001 
        self.beta = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Decision_transformer(s_dim=self.s_dim, a_dim=self.a_dim, hidden_dim=64, K=self.K, max_timestep=self.max_timestep).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters())
        
        self.character = enums.Character.FOX
        self.action_space = MyActionSpace()
        self.observation_space = obs_space
            
        #buffers & temporal parameters
        self.timestep = 0
        self.last_r = None
        self.traj_buffer = TrajBuffer(K = self.K, buffer_size = self.buffer_size, a_dim=self.a_dim, device=self.device)
        
        #temporal buffers : reset after each episodes
        self.temp_buffer = {'R' : None,
                            's' : None,
                            'a' : None }

        #turn on eval mode during exploration
        self.model.eval()
    
    @from_action_space
    @from_observation_space
    def act(self, gamestate):
        obs, reward, done, info = gamestate
        self.push_R(reward)
        self.push_s(obs)
        self.last_r = reward
        R, s, a= self.get_Rsa_seq()
        t = (torch.arange(self.timestep-self.K+1, self.timestep + 1) if self.timestep-self.K>0 else torch.arange(0, self.K)).unsqueeze(0).to(device=self.device)
        _, _, a_preds = self.model.forward(t=t, R=R, s=s, a=a)
        self.timestep += 1

        a_pred = Categorical(probs=a_preds[0, min(self.K - 1, self.timestep)]).sample().detach()
        self.push_a(a_pred)
        return a_pred
        
    def train(self):
        print("training...")
        self.model.train()
        for _ in tqdm(range(self.training_iter)):
            self.optimizer.zero_grad()
            T, R, s, a, masks = self.traj_buffer.get_trajs(self.batch_size)
            _, _, a_preds = self.model.forward(T, R, s, a)
            L, dlmbda = self.lagrangian(a, a_preds, masks, lmbda=self.model.lmbda) #to ensure its shannon entropy is larger than beta, a fixed const,
            L.backward() #we optimize Lagrangian associated with both loss fn and entropy.
            self.optimizer.step() #optimizing theta, parameters of the model
            self.model.lmbda = max(self.model.lmbda + self.lmbda_lr * dlmbda, 0) #optimizing lmbda
            print("loss : ", L)
        self.model.eval()
        print("done!")

    def end_ep(self):
        self.update_traj_buffer()
        self.timestep = 0
        self.last_r = None
        for x in self.temp_buffer.keys():
            self.temp_buffer[x] = None


    
    def loss_fn(self, a, a_preds, masks):
        return torch.sum(- a * torch.log(a_preds) * (1 - masks))

    def lagrangian(self, a, a_preds, masks, lmbda):
        J = self.loss_fn(a, a_preds, masks) / self.K
        H = torch.sum(Categorical(probs = a_preds).entropy()) / self.K
        H_ = - H + self.beta
        return J + lmbda * H_, H_.item()


    def update_traj_buffer(self):
        R = self.temp_buffer['R'].clone().to('cpu')
        s = self.temp_buffer['s'].clone().to('cpu')        
        a = self.temp_buffer['a'].clone().to('cpu')
        R = R - R[-1] + self.last_r #Hindsight return relabeling
        self.traj_buffer.push_traj(R, s, a)
        

    def push_R(self, r):
        if self.temp_buffer['R'] is None:
            self.temp_buffer['R'] = torch.tensor([[self.objective_R]]).to(self.device, dtype=torch.float32)
        else:
            RtG = self.temp_buffer['R'][-1].item() - r
            self.temp_buffer['R'] = torch.cat((self.temp_buffer['R'], torch.tensor([[RtG]]).to(self.device, dtype=torch.float32)), dim=0)
    
    def push_s(self, s):
        if self.temp_buffer['s'] is None:
            s = torch.tensor(state_processor(s))
            self.temp_buffer['s'] = s.unsqueeze(0).to(self.device, dtype=torch.float32)
        else:
            self.temp_buffer['s'] = torch.cat((self.temp_buffer['s'], torch.tensor(state_processor(s)).unsqueeze(0).to(self.device, dtype=torch.float32)), dim=0)

    def push_a(self, a):
        if self.temp_buffer['a'] is None:
            self.temp_buffer['a'] = a.unsqueeze(0).to(self.device)
        else:
            self.temp_buffer['a'] = torch.cat((self.temp_buffer['a'], a.unsqueeze(0).to(self.device)), dim=0)

    def add_padding(self, seq):
        T, h = seq.shape
        if self.K - T > 0:
            seq = torch.cat((seq, torch.zeros(self.K - T, h).to(self.device)), dim=0)
        return seq
    
    def get_Rsa_seq(self):
        i = max(self.timestep - self.K + 1, 0)
        R = self.add_padding(self.temp_buffer['R'][i:-1]).unsqueeze(0)
        s = self.add_padding(self.temp_buffer['s'][i:-1]).unsqueeze(0)
        if self.temp_buffer['a'] is not None:
            a = self.add_padding(F.one_hot(self.temp_buffer['a'][i - 1:-1], num_classes=self.a_dim)).unsqueeze(0)
        else:
            a = torch.zeros(self.K, self.a_dim).unsqueeze(0).to(self.device)
        return [x.to(dtype=torch.float32) for x in [R, s, a]]