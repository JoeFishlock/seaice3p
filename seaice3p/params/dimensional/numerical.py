from dataclasses import dataclass
from serde import serde, coerce


@serde(type_check=coerce)
@dataclass(frozen=True)
class NumericalParams:
    """parameters needed for discretisation and choice of numerical method"""

    I: int = 50
    regularisation: float = 1e-6
    solver_choice: str = "RK23"  # scipy.integrate.solve_IVP solver choice

    @property
    def step(self):
        return 1 / self.I
