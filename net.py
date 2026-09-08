# -*- coding: utf-8 -*-
import numpy as np
import torch.nn as nn


def layer_init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    """正交初始化线性层权重，偏置置零。"""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Soft_Qnet(nn.Module):

    def __init__(self, state_dim: int, action_dim: int, hid_dim: int=256):
        super().__init__()

        self.feature_layer = nn.Sequential(
            layer_init(nn.Linear(state_dim, hid_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hid_dim, hid_dim)),
            nn.ReLU()
        )
        self.value_head = layer_init(nn.Linear(hid_dim, 1), gain=0.01)
        self.advantage_head = layer_init(nn.Linear(hid_dim, action_dim), gain=0.01)

    def forward(self, state):
        features = self.feature_layer(state)
        v = self.value_head(features)
        advan = self.advantage_head(features)

        q = v + (advan - advan.mean(dim=1, keepdim=True))
        return q

