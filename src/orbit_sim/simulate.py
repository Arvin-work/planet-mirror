import numpy as np
from .diagnostics import specific_angular_momentum, specific_energy

def simulate_orbit(position, velocity, dt, steps, integrator):
    """
    Simulate one orbit using one numerical integrator.

    Parameters
    ----------
    position : np.ndarray
        Initial position.

    velocity : np.ndarray
        Initial velocity.

    dt : float
        Time step.

    steps : int
        Number of integration steps.

    integrator : callable
        Numerical integrator.

    Returns
    -------
    dict
        position, velocity, energy and angular momentum history.
    """

    position_using = position.copy()
    velocity_using = velocity.copy()

    positions = [
        position.copy()
    ]
    velocities = [
        velocity.copy()
    ]
    energies = [
        specific_energy(position=position_using, velocity=velocity_using)
    ]
    anagular_momenta = [
        specific_angular_momentum(
            position=position,
            velocity=velocity
        )
    ]

    for _ in range(steps):
        position_using, velocity_using = integrator(
            position = position_using,
            velocity = velocity_using,
            dt = dt
        )

        positions.append(position_using.copy())
        velocities.append(velocity_using.copy())
        energies.append(specific_energy(position=position_using, velocity=velocity_using))
        anagular_momenta.append(specific_angular_momentum(position=position_using, velocity=velocity_using))

    return {
            "positions": np.array(positions),
            "velocities": np.array(velocities),
            "energy": np.array(energies),
            "anguluar_momentum": np.array(anagular_momenta)
    }

def simulate_orbit_with_multiple_integrators(position, velocity, dt, steps, integrators):
    """
    Simulate one orbit using multiple numerical integrators.

    Parameters
    ----------
    position : np.ndarray
        Initial position.

    velocity : np.ndarray
        Initial velocity.

    dt : float
        Time step.

    steps : int
        Number of integration steps.

    integrators : list of callable
        List of numerical integrators.

    Returns
    -------
    dict
        Dictionary containing the results for each integrator.
    """

    results = {}

    for name, integrator in integrators.items():
        result = simulate_orbit(position, velocity, dt, steps, integrator)
        results[name] = result

    return results


def simulate_nbody(positions, velocities, masses, dt, steps, integrator,
                   gravitating=None, acceleration_function=None):
    """Synchronously propagate all rows; histories include the initial snapshot.

    Custom force must be position-only. Diagnostics describe Newtonian active
    bodies; additional forces can invalidate their conservation.
    """
    from .dynamics import _system_arrays, make_nbody_acceleration
    from .diagnostics import (total_energy, total_linear_momentum,
                              total_angular_momentum, center_of_mass)
    r, m, mask = _system_arrays(positions, masses, gravitating)
    r, m, mask = r.copy(), m.copy(), mask.copy()
    v = np.asarray(velocities, dtype=float).copy()
    if v.shape != r.shape or not np.all(np.isfinite(v)):
        raise ValueError("velocities must be finite and match positions")
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    force = acceleration_function if acceleration_function is not None else make_nbody_acceleration(m, mask)
    history = {key: [] for key in ('positions', 'velocities', 'total_energy',
                                  'total_linear_momentum', 'total_angular_momentum', 'center_of_mass')}
    def record():
        history['positions'].append(r.copy())
        history['velocities'].append(v.copy())
        history['total_energy'].append(total_energy(r, v, m, mask))
        history['total_linear_momentum'].append(total_linear_momentum(v, m, mask))
        history['total_angular_momentum'].append(total_angular_momentum(r, v, m, mask))
        history['center_of_mass'].append(center_of_mass(r, m, mask) if np.sum(m[mask]) > 0 else np.full(3, np.nan))
    positions_shape = r.shape
    record()
    for _ in range(steps):
        r, v = integrator(r, v, dt, acceleration_function=force)
        r, v = np.asarray(r, dtype=float), np.asarray(v, dtype=float)
        if r.shape != positions_shape or v.shape != positions_shape:
            raise ValueError("integrator changed system state shape")
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(v)):
            raise ValueError("integrator returned nonfinite state")
        record()
    result = {key: np.asarray(value) for key, value in history.items()}
    result['times'] = np.arange(steps + 1) * dt
    return result
