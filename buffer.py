import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random

class TrajBuffer():
    def __init__(self, K, buffer_size, a_dim):
        self.K = K
        self.buffer_size = buffer_size
        self.a_dim = a_dim
        self.R = []
        self.s = []
        self.a = []
        self.len_list = []

    def push_traj(self, R, s, a):
        self.R.append(R)
        self.s.append(s)
        self.a.append(a)
        self.len_list.append(R.shape[0])
        if len(self.len_list) > self.buffer_size:
            self.R = self.R[1:]
            self.s = self.s[1:]
            self.a = self.a[1:]
            self.len_list = self.len_list[1:]

    def add_padding(self, seq, give_mask = False):
        _, S, h = seq.shape
        if self.K - S > 0:
            seq = torch.cat(seq, torch.zeros(1, self.K - S, h), dim=1)

        if give_mask:
            mask = torch.zeros(self.K)
            mask[self.S - 1:] = 1
            return seq, mask
        else:
            return seq

    def get_trajs(self, batch_size):
        T = []
        R = []
        s  = []
        a = []
        for _ in range(batch_size):
            i = Categorical(self.len_list).sample(sample_shape=(1,)).item()
            t = random() % (self.len_list[i] - self.K + 1)
            t_ = t + self.K
            T.append(torch.arange(t, t_).unsqueeze(0))
            R.append(self.R[t : t_])
            s.append(self.s[t : t_])
            a.append(self.a[t : t_])

        for x in [T, R, s, a]:
            x = torch.stack(x)
        return (T, R, s, a)
        
        
