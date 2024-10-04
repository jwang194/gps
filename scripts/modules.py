import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

class PopStruct(nn.Module):
    def __init__(self, n_layers, input_dim, rep_dim, output_dim):
        super(PopStruct, self).__init__()
        midpoint = n_layers//2
        up_scale = round(np.power(rep_dim//input_dim,1/midpoint))
        down_scale = round(np.power(rep_dim//output_dim,1/midpoint))
        self.dense_layers = nn.ModuleList()
        for i in range(n_layers):
            if i < midpoint:
                self.dense_layers.append(nn.Linear(input_dim,input_dim*up_scale))
                input_dim *= up_scale
            else:   
                self.dense_layers.append(nn.Linear(rep_dim,rep_dim//down_scale))
                rep_dim //= down_scale

    def forward(self, x):
        for i in range(len(self.dense_layers)-1):
            x = F.relu(self.dense_layers[i](x))
        return self.dense_layers[-1](x)
