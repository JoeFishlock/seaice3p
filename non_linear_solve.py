import time
import numpy as np
from scipy.optimize import root


def calculate_residual(new_solution, solution):
    new_enthalpy, new_salt, new_gas, new_pressure = np.split(new_solution, 4)
    enthalpy, salt, gas, pressure = np.split(solution, 4)
    enthalpy_residual = new_enthalpy**3 - new_salt
    salt_residual = new_salt - salt
    gas_residual = 4 * new_gas - gas
    pressure_residual = new_enthalpy * new_salt - new_pressure - pressure
    return np.hstack(
        (enthalpy_residual, salt_residual, gas_residual, pressure_residual)
    )


def solve_non_linear_system(solution):
    new_solution = root(
        lambda x: calculate_residual(x, solution),
        x0=solution,
        method="krylov",
    )
    return new_solution


enthalpy = np.linspace(1, 2, 200)
salt = np.geomspace(2, 4, 200)
gas = np.linspace(-6, 10, 200)
pressure = np.geomspace(1, 10, 200)
solution = np.hstack((enthalpy, salt, gas, pressure))
solution = 0.66 + solution**2
t0 = time.time()
new_solution = solve_non_linear_system(solution)
t1 = time.time()
print(new_solution.x)
print(new_solution.message)
print(t1 - t0)
