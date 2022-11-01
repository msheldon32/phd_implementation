import torch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class AltDQN(nn.Module):
    """
        Alternative formulation of a deep Q network, with one "default" parameter and the rest representing deltas
    """
    def __init__(self, n_inputs, hidden_size, n_outputs):
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_outputs)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.stack(x)


class SubsidyAgent:
    def __init__(self, n_cells):
        # state: 
        self.network = DQN(n_cells+1, n_cells+1, n_cells+1)
        self.eps_max = 0.9
        self.eps_min = 0.05
        self.eps_decay = 200
        self.gamma = 0.99
        self.total_steps_taken = 0
    
    def select_action(self, state):
