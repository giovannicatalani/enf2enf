from enf.steerable_attention.invariant._base_invariant import BaseInvariant
from enf.steerable_attention.invariant.rel_pos import RelativePositionND


def get_sa_invariant(cfg) -> BaseInvariant:
    """ Get the invariant for the self attention module.

        Args:
            name (str): The name of the invariant.

        Returns:
            BaseInvariant: The invariant module.

        """
    
    if cfg.invariant_type == "rel_pos":
        return RelativePositionND(num_dims=cfg.num_in)
    else:
        raise ValueError(f"Unknown invariant type: {cfg.invariant_type}.")


def get_ca_invariant(cfg) -> BaseInvariant:
    """ Get the invariant for the cross attention module.

    Args:
        name (str): The name of the invariant.

    Returns:
        BaseInvariant: The invariant module.

    """
    if cfg.invariant_type == "rel_pos":
        return RelativePositionND(num_dims=cfg.num_in)
    else:
        raise ValueError(f"Unknown invariant type: {cfg.invariant_type}.")
