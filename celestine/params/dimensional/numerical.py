from serde import serde, coerce


@serde(type_check=coerce)
class NumericalParams:
    """parameters needed for discretisation and choice of numerical method"""

    I: int = 50
    regularisation: float = 1e-6

    @property
    def step(self):
        return 1 / self.I
