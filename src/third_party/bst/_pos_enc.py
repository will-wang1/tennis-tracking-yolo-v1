"""Stand-in for `positional_encodings.torch_encodings.PositionalEncoding1D`
(the only thing BST used from that package): sinusoidal encodings for a
(batch, x, ch) tensor. BST only uses it to INITIALISE its positional
embeddings, which a trained checkpoint then overwrites, so inference does
not depend on it matching the package exactly."""

import numpy as np
import torch


class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels)))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, x, _ = tensor.shape
        pos = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp = torch.einsum("i,j->ij", pos, self.inv_freq)
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1).flatten(-2, -1)
        return emb[None, :, : self.org_channels].repeat(batch, 1, 1).to(tensor.dtype)
