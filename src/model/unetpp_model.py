import torch
import torch.nn as nn
from ..blocks.encoder_block import EncoderBlock
from ..blocks.decoder_block import DecoderBlock
from ..blocks.nested_skip_unit import NestedSkipUnit
from ..layers.deep_supervision_head import DeepSupervisionHead

class UNetPP(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, num_classes=1, depth=4):
        super().__init__()
        self.depth = depth
        # Encoder
        self.encoders = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out_ch = base_channels * (2**i)
            self.encoders.append(EncoderBlock(channels, out_ch))
            channels = out_ch

        # Decoder / Nested skip placeholders
        self.nested_skips = nn.ModuleDict()
        # Deep supervision heads
        self.deep_heads = nn.ModuleList([DeepSupervisionHead(base_channels*(2**i), num_classes) for i in range(1, depth+1)])

    def forward(self, x):
        encoder_outputs = []
        x_in = x
        for enc in self.encoders:
            x_skip, x_in = enc(x_in)
            encoder_outputs.append(x_skip)
        deep_outputs = [head(enc_out) for head, enc_out in zip(self.deep_heads, encoder_outputs[1:])]
        return deep_outputs
