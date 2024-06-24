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
import math

from hyper_param import PARAMS

class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x

class Decision_transformer(nn.Module):
    def __init__(self, s_dim, a_dim, hidden_dim, K, max_timestep): #the model will consider last K timesteps
        super(Decision_transformer, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.hidden_dim = hidden_dim
        self.K = K
        self.embed_t = nn.Embedding(max_timestep, hidden_dim)
        self.embed_R = nn.Linear(1, hidden_dim)
        self.embed_s = nn.Linear(s_dim, hidden_dim)
        self.embed_a = nn.Linear(a_dim, hidden_dim)
        self.embed_layernorm = nn.LayerNorm(hidden_dim)
        self.transformer = nn.Sequential(
            Block(hidden_dim, 3 * K, n_heads=4),
            Block(hidden_dim, 3 * K, n_heads=4),
            Block(hidden_dim, 3 * K, n_heads=4),
            Block(hidden_dim, 3 * K, n_heads=4)
        )
        self.predict_R = nn.Linear(hidden_dim, 1)
        self.predict_s = nn.Linear(hidden_dim, s_dim)
        self.predict_a = nn.softmax(nn.Linear(hidden_dim, a_dim)) #original paper doesn't apply softmax()

    def forward(self, t, R, s, a): #(Batch, Seq, *)
        B, T, _ = s.shape
        t_embedding = self.embed_t()
        R_embedding = self.embed_R(R) + t_embedding
        s_embedding = self.embed_s(s) + t_embedding
        a_embedding = self.embed_a(a) + t_embedding

        h = torch.stack(
            (R_embedding, s_embedding, a_embedding), dim=1
        ).permute(0, 2, 1, 3).reshape(B, 3 * T, self.h_dim)
        h = self.embed_layernorm(h)
        h = self.transformer(h)

        # get h reshaped such that its size = (B x 3 x T x h_dim) and
        # h[:, 0, t] is conditioned on r_0, s_0, a_0 ... r_t
        # h[:, 1, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t
        # h[:, 2, t] is conditioned on r_0, s_0, a_0 ... r_t, s_t, a_t
        h = h.reshape(B, T, 3, self.h_dim).permute(0, 2, 1, 3)

        # get predictions
        R_preds = self.predict_R(h[:,2])     # predict next rtg given r, s, a
        s_preds = self.predict_s(h[:,2])    # predict next state given r, s, a
        a_preds = self.predict_a(h[:,1])  # predict action given r, s
    
        return R_preds, s_preds, a_preds
    




        


