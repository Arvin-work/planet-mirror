import math
import numpy as np

G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_EARTH = 5.972e24  # Mass of Earth in kg

def gravitational_acceleration(point_position: np.ndarray, earth_position: np.ndarray = np.array([0.0, 0.0, 0.0])):
    """
    Calculate the gravitational acceleration at a given position in space.

    Parameters:
    position (numpy.ndarray): A 3D vector representing the position in meters.

    Returns:
    numpy.ndarray: A 3D vector representing the gravitational acceleration in m/s^2.
    """

    displacement_vector = point_position - earth_position
    r = np.linalg.norm(point_position - earth_position)  # Distance from the point to the center of the Earth

    if r == 0:
        raise ValueError("Position cannot be at the center of the Earth.")

    a = -G * M_EARTH * displacement_vector / r**3  # Magnitude of gravitational acceleration
    return a


def _system_arrays(positions, masses, gravitating=None):
    """Validate a snapshot; masks select sources, not propagated targets."""
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if masses.shape != (len(positions),):
        raise ValueError("masses must have shape (N,)")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(masses)):
        raise ValueError("positions and masses must be finite")
    if np.any(masses < 0):
        raise ValueError("masses must be nonnegative")
    if gravitating is None:
        gravitating = np.ones(len(masses), dtype=bool)
    else:
        gravitating = np.asarray(gravitating)
        if gravitating.shape != masses.shape or gravitating.dtype != np.dtype(bool):
            raise ValueError("gravitating must be a boolean array of shape (N,)")
    return positions, masses, gravitating


def gravitational_acceleration_from_sources(target_position, source_positions, source_masses):
    """Return (3,) acceleration in SI units from point sources (M,3)/(M,).

    Zero-mass sources are ignored. Positive-mass coincident sources raise;
    callers must explicitly exclude self, never exclude by position alone.
    """
    target = np.asarray(target_position, dtype=float)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_position must be a finite (3,) vector")
    sources, masses, _ = _system_arrays(source_positions, source_masses)
    displacement = sources[masses > 0] - target
    radius = np.linalg.norm(displacement, axis=1)
    if np.any(radius == 0):
        raise ValueError("Distinct target and positive-mass source coincide")
    return np.sum(G * masses[masses > 0, None] * displacement / radius[:, None]**3, axis=0)


def nbody_accelerations(positions, masses, gravitating=None):
    """Return (N,3) accelerations for one synchronized snapshot.

    All rows receive acceleration; only gravitating=True rows are sources.
    Self interaction is excluded by row index. No softening is applied.
    """
    positions, masses, mask = _system_arrays(positions, masses, gravitating)
    acceleration = np.zeros_like(positions)
    indices = np.arange(len(masses))
    for i, target in enumerate(positions):
        sources = mask & (indices != i)
        acceleration[i] = gravitational_acceleration_from_sources(
            target, positions[sources], masses[sources])
    return acceleration


def make_nbody_acceleration(masses, gravitating=None):
    """Freeze source metadata and return acceleration(positions) for integrators."""
    masses = np.asarray(masses, dtype=float)
    if masses.ndim != 1:
        raise ValueError("masses must have shape (N,)")
    _, masses, mask = _system_arrays(np.zeros((len(masses), 3)), masses, gravitating)
    masses, mask = masses.copy(), mask.copy()

    def acceleration(positions):
        return nbody_accelerations(positions, masses, mask)

    return acceleration
