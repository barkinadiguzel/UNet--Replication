import torch
import torch.nn as nn
from ..layers.dense_skip_block import DenseSkipBlock
from ..layers.upsample_layer import UpSampleLayer

class NestedSkipUnit(nn.Module):
    def __init__(self, in_channels_list, growth_rate, num_layers):
        super().__init__()
        self.dense_block = DenseSkipBlock(sum(in_channels_list), growth_rate, num_layers)

    def forward(self, x_list):
        return self.dense_block(x_list)
