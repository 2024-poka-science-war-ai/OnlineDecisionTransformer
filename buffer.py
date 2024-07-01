import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import random

class TrajBuffer():
    def __init__(self, K, buffer_size, a_dim, device):
        self.device = device
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
        S, h = seq.shape
        if self.K - S > 0:
            seq = torch.cat((seq, torch.zeros(self.K - S, h)), dim=0)
        if give_mask:
            mask = torch.zeros((self.K,))
            mask[S:] = 1
            return seq, mask.unsqueeze(1)
        else:
            return seq

    def get_trajs(self, batch_size):
        T = []
        R = []
        s  = []
        a = []
        masks = []
        for _ in range(batch_size):
            i = Categorical(torch.tensor(self.len_list)).sample(sample_shape=(1,)).item()
            t = random.randrange(0, self.len_list[i])
            t_ = min(t + self.K, self.len_list[i])
            T.append(torch.arange(t, t + self.K))
            R.append(self.add_padding(self.R[i][t : t_]).to(dtype=torch.float32))
            s.append(self.add_padding(self.s[i][t : t_]).to(dtype=torch.float32))
            action_seq, mask = self.add_padding(F.one_hot(self.a[i][t : t_], num_classes=self.a_dim).to(dtype=torch.float32), give_mask=True)
            a.append(action_seq)
            masks.append(mask)

        traj = [T, R, s, a, masks]
        for i in range(5):
            traj[i] = torch.stack(traj[i]).to(self.device)
        return traj
        
        
