import jax.numpy as jnp
from flax import linen as nn
from functools import partial

#Code adapted from original implementation of ENFs implementation https://github.com/david-knigge/enf-pde/tree/main

def init_positions_grid(rng, shape, spatial_dims=None, bbox=None):
    """
    Initialize the latent poses on a grid using JAX, with optional specification of spatial dimensions
    and bounding box.

    Args:
        rng: JAX random key (not used in grid initialization but kept for API consistency)
        shape: tuple (num_signals, num_latents, num_dims) - The desired shape of the output
        spatial_dims: int or None - Number of dimensions to create grid for. If None, uses all dimensions.
                        If specified, creates grid for first spatial_dims dimensions and sets others to 0.
        bbox: tuple or None - The bounding box for the grid points. If None, uses the default range [-1, 1] for each dimension.

    Returns:
        z_positions (jax.numpy.ndarray): The latent poses for each signal. 
        Shape [num_signals, num_latents, num_dims].
    """
    num_signals, num_latents, num_dims = shape
    spatial_dims = spatial_dims or num_dims  # If not specified, use all dimensions

    # Ensure num_latents is a power of spatial_dims
    assert abs(round(num_latents ** (1. / spatial_dims), 5) % 1) < 1e-5, \
        f'num_latents ({num_latents}) must be a power of the number of spatial dimensions ({spatial_dims})'

    # Calculate the number of latents per dimension
    num_latents_per_dim = int(round(num_latents ** (1. / spatial_dims)))

    if bbox is None:
        # Create mesh grid for spatial dimensions with default range [-1, 1]
        grid_axes = jnp.linspace(-1 + 1/num_latents_per_dim, 1 - 1/num_latents_per_dim, num_latents_per_dim)
        grids = jnp.meshgrid(*[grid_axes for _ in range(spatial_dims)], indexing='ij')

    else:
        # Create mesh grid for spatial dimensions with specified bounding box
        grid_axes = [jnp.linspace(bbox[i][0], bbox[i][1], num_latents_per_dim) for i in range(spatial_dims)]
        grids = jnp.meshgrid(*grid_axes, indexing='ij')

    # Stack and reshape to create the positions matrix for spatial dimensions
    spatial_positions = jnp.stack(grids, axis=-1).reshape(-1, spatial_dims)

    # If we have additional non-spatial dimensions, pad with zeros
    if num_dims > spatial_dims:
        zeros = jnp.zeros((spatial_positions.shape[0], num_dims - spatial_dims))
        positions = jnp.concatenate([spatial_positions, zeros], axis=-1)
    else:
        positions = spatial_positions

    # Repeat for the number of signals
    positions = jnp.repeat(positions[None, :, :], num_signals, axis=0)

    return positions

    

class PositionOrientationFeatureAutodecoder(nn.Module):
    num_signals: int
    num_latents: int
    latent_dim: int
    num_pos_dims: int
    num_ori_dims: int
    gaussian_window_size: float = None
    frequency_parameter: float = None
    coordinate_system: str = 'cartesian'
    spatial_dims: int = None  # Only used for cartesian grid initialization
    bbox: tuple = None  # Only used for cartesian grid initialization


    def setup(self):
        # Initialize the latent positions, orientations, appearances, gaussian window, and frequency parameter here
        if self.coordinate_system == 'cartesian':
            self.p_pos = self.param(
                'p_pos', 
                lambda rng, shape: init_positions_grid(
                    rng, 
                    shape, 
                    spatial_dims=self.spatial_dims,
                    bbox=self.bbox
                ), 
                (self.num_signals, self.num_latents, self.num_pos_dims)
            )
        

        self.a = self.param('a', nn.initializers.ones, (self.num_signals, self.num_latents, self.latent_dim))

        # Calculate gaussian window size s.t. each gaussian overlaps.
        # This is the same as setting the standard deviation to the distance between the latent points.
        # Create a grid of latent positions
        if self.coordinate_system == 'cartesian':
            spatial_dims = self.spatial_dims or self.num_pos_dims
            num_latents_per_dim = int(round(self.num_latents ** (1. / spatial_dims), 5))

            # Since our domain ranges from -1 to 1, the distance between each latent point is 2 / num_latents_per_dim
            # We want each gaussian to be centered at a latent point, and be removed 2 std from other latent points.
            gaussian_window_size = self.num_pos_dims / num_latents_per_dim

        self.gaussian_window = self.param('gaussian_window', nn.initializers.constant(gaussian_window_size), (self.num_signals, self.num_latents, 1))

    def __call__(self, idx: int):
        # Implement the forward pass using JAX operations
        p_pos = self.p_pos[idx]

        if self.num_ori_dims > 0:
            p_ori = self.p_ori[idx]
            p = jnp.concatenate((p_pos, p_ori), axis=-1)
        else:
            p = p_pos

        a = self.a[idx]

        # Optionally, get the gaussian window for the latent points
        gaussian_window = self.gaussian_window[idx]

        return p, a, gaussian_window
    
class PositionOrientationFeatureAutodecoderMeta(PositionOrientationFeatureAutodecoder):

    def __call__(self):
        # Implement the forward pass using JAX operations
        p_pos = self.p_pos

        if self.num_ori_dims > 0:
            p_ori = self.p_ori
            p = jnp.concatenate((p_pos, p_ori), axis=-1)
        else:
            p = p_pos

        a = self.a

        # Optionally, get the gaussian window for the latent points
        if self.gaussian_window_size is not None:
            gaussian_window = self.gaussian_window
        else:
            gaussian_window = None
        return p, a, gaussian_window
    

