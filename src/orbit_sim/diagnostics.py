import numpy as np

G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
M_Earth = 5.972e24  # Mass of Earth in kg

def specific_energy(position, velocity):
    """
    Calculate the specific energy of the system.
    """
    r = np.linalg.norm(position)
    if r == 0:
        raise ValueError("Position cannot be at the center of the Earth.")

    return 0.5 * np.linalg.norm(velocity)**2 - G * M_Earth / np.linalg.norm(position)
def specific_angular_momentum(position, velocity):
    """
    Calculate the specific angular momentum of the system.
    """
    return np.cross(position, velocity)


from .dynamics import _system_arrays


def _selected(positions, velocities, masses, gravitating):
    positions, masses, mask = _system_arrays(positions, masses, gravitating)
    velocities = np.asarray(velocities, dtype=float)
    if velocities.shape != positions.shape or not np.all(np.isfinite(velocities)):
        raise ValueError("velocities must be finite and match positions")
    mask = mask & (masses > 0)
    return positions[mask], velocities[mask], masses[mask]


def total_energy(positions, velocities, masses, gravitating=None):
    """Kinetic + pair potential energy (J), for selected active bodies only."""
    r, v, m = _selected(positions, velocities, masses, gravitating)
    energy = 0.5 * np.sum(m[:, None] * v**2)
    for i in range(len(m)):
        distances = np.linalg.norm(r[i+1:] - r[i], axis=1)
        if np.any(distances == 0):
            raise ValueError("Positive-mass bodies coincide")
        energy -= np.sum(G * m[i] * m[i+1:] / distances)
    return energy


def total_linear_momentum(velocities, masses, gravitating=None):
    """Return (3,) momentum in kg m/s for selected active bodies."""
    v = np.asarray(velocities, dtype=float)
    _, v, m = _selected(np.zeros_like(v), v, masses, gravitating)
    return np.sum(m[:, None] * v, axis=0)


def total_angular_momentum(positions, velocities, masses, gravitating=None):
    """Return (3,) angular momentum about the inertial origin (kg m²/s)."""
    r, v, m = _selected(positions, velocities, masses, gravitating)
    return np.sum(m[:, None] * np.cross(r, v), axis=0)


def center_of_mass(positions, masses, gravitating=None):
    """Return (3,) active center of mass (m); require positive selected mass."""
    r, m, mask = _system_arrays(positions, masses, gravitating)
    if np.sum(m[mask]) <= 0:
        raise ValueError("Center of mass requires positive selected total mass")
    return np.sum(m[mask, None] * r[mask], axis=0) / np.sum(m[mask])
