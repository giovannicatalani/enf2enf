from typing import Any, Dict
import numpy as np
import torch
import torch.nn as nn
try:
    from utils_functa.conditioning import shift_modulation,cross_attention_conditioning, CrossAttentionModulation
except:
    from conditioning import shift_modulation,cross_attention_conditioning, CrossAttentionModulation

class LatentToModulation(nn.Module):
    """Maps a latent vector to a set of modulations.
    Args:
        latent_dim (int):
        num_modulations (int):
        dim_hidden (int):
        num_layers (int):
    """

    def __init__(
        self, latent_dim, num_modulations, dim_hidden, num_layers, activation=nn.SiLU
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modulations = num_modulations
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.activation = activation

        if num_layers == 1:
            self.net = nn.Linear(latent_dim, num_modulations)
        else:
            layers = [nn.Linear(latent_dim, dim_hidden), self.activation()]
            if num_layers > 2:
                for i in range(num_layers - 2):
                    layers += [nn.Linear(dim_hidden, dim_hidden),
                               self.activation()]
            layers += [nn.Linear(dim_hidden, num_modulations)]
            self.net = nn.Sequential(*layers)

    def forward(self, latent):
        return self.net(latent)
    

class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        """

        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )

class ModulatedFourierFeatures(nn.Module):
    """WARNING: the code does not support non-graph inputs.
        It needs to be adapted for (batch, num_points, coordinates) format
        The FiLM Modulated Network with Fourier Embedding used for the experiments on Airfrans.
        The code relies on conditoning functions: film, film_linear and film_translate.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        num_frequencies=8,
        latent_dim=128,
        width=256,
        depth=3,
        depth_hnn = 1,
        modulate_scale=False,
        modulate_shift=True,
        frequency_embedding="nerf",
        include_input=True,
        scale=5,
        max_frequencies=32,
        base_frequency=1.25,
    ):
        super().__init__()
        self.frequency_embedding = frequency_embedding
        self.include_input = include_input
        
        self.scale = scale
        self.embedding = GaussianEncoding(
            embedding_size=num_frequencies * 2, scale=scale, dims=input_dim
        )
        embed_dim = (
            num_frequencies * 2 + input_dim
            if include_input
            else num_frequencies * 2
        )
        self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [output_dim]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.depth = depth
        self.hidden_dim = width
        self.depth_hnn = depth_hnn
        self.num_modulations = self.hidden_dim * (self.depth - 1)
        if modulate_scale and modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            self.num_modulations *= 2
        self.latent_to_modulation = LatentToModulation(
            self.latent_dim, self.num_modulations, dim_hidden=128, num_layers=self.depth_hnn
        )
        
        self.conditioning = shift_modulation

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        #print('x', x.shape, z.shape)
        features = self.latent_to_modulation(z)
        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], axis=-1)
        pre_out = self.conditioning(position, features, self.layers[:-1], torch.relu)
        out = self.layers[-1](pre_out)
        return out.view(*x_shape, out.shape[-1])



    