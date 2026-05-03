"""
the new relational layer implementation that uses fully vectorized tensor operations
instead of O(K²) Python loops.

- final implementation used in the paper's RAER model: @relational_layer.py
- old implementation with O(K²) loops (not used in final model): @old_relationl_layer.py
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape

        xi = x.unsqueeze(2).expand(b, n, n, d)  # (b, n, 1, d) -> (b, n, n, d)
        xj = x.unsqueeze(1).expand(b, n, n, d)  # (b, 1, n, d) -> (b, n, n, d)
        pairs = torch.cat([xi, xj], dim=-1)       # (b, n, n, 2d)

        x = self.fc1(pairs)
        x = self.relu(self.fc2(x))

        mask = ~torch.eye(n, dtype=torch.bool, device=x.device) 
        x = x * mask[None, :, :, None] 
        
        # sum over neighbors
        return x.sum(dim=2) # (b, n, out_dim)
