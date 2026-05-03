"""
old implementation of relational layer, not used in final model.
the cons is that it uses O(K²) Python loops which is very slow for K=12 person graphs.

this file is kept for reference and comparison
to the final vectorized implementation in @relational_layer.py
 -> old_relationl_layer.py - V1

"""

import torch 
import torch.nn as nn

class RelationalLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim
        
        self.fc1 = nn.Linear(in_dim * 2, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        b, n, d = x.shape
        for i in range(n):
            sum_neighbors = torch.zeros(b, self.out_dim, device=x.device)
            for j in range(n):
                if i != j:
                    pair = torch.cat([x[:, i, :], x[:, j, :]], dim=-1) # (b, 2d)
                    h = self.fc1(pair) # (b, out_dim)
                    h = self.relu(self.fc2(h)) # (b, out_dim)
                    sum_neighbors += h
            x[:, i, :] = sum_neighbors
        return x

    