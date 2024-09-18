from ...params import Config


def calculate_bubble_size_fraction(bubble_radius_scaled, liquid_fraction, cfg: Config):
    r"""Takes bubble radius scaled and liquid fraction on edges and calculates the
    bubble size fraction as

    .. math:: \lambda = \Lambda / (\phi_l^q + \text{reg})

    Returns the bubble size fraction on the edge grid.
    """
    exponent = cfg.bubble_params.pore_throat_scaling
    reg = cfg.numerical_params.regularisation
    effective_tube_radius = liquid_fraction**exponent + reg
    return bubble_radius_scaled / effective_tube_radius
