import numpy as np
from .dynamics import G, M_EARTH, gravitational_acceleration
from .diagnostics import specific_energy, specific_angular_momentum

def circular_orbit_velocity(
    position: np.ndarray,
    velocity_direction: np.ndarray
) -> np.ndarray:
    """
    Calculate the velocity required for a circular orbit.

    Parameters
    ----------
    position : np.ndarray
        Initial position vector.

    velocity_direction : np.ndarray
        Direction of the initial velocity.

    Returns
    -------
    np.ndarray
        Circular-orbit velocity vector.
    """

    radius = np.linalg.norm(position)
    if radius == 0:
        raise ValueError(
            "Position cannot be at the center."
        )
    direction_norm = np.linalg.norm(velocity_direction)
    if direction_norm == 0:
        raise ValueError("Velocity direction cannot be zero.")
    direction = (velocity_direction / direction_norm)

    # Circular velocity must be perpendicular
    # to the radius vector.
    radial_direction = (position / radius)
    perpendicularity = np.dot(radial_direction, direction)
    if not np.isclose(perpendicularity, 0.0):
        raise ValueError(
            "Velocity direction must be "
            "perpendicular to position "
            "for a circular orbit."
        )
    velocity_magnitude = np.sqrt(G * M_EARTH / radius)
    velocity = (
        velocity_magnitude
        * direction
    )

    return velocity

def circular_orbit_period(radius):
    """
    Calculate the orbital period for a circular orbit at a given radius.

    Parameters:
    radius (float): The radius of the circular orbit in meters.

    Returns:
    float: The orbital period in seconds.
    """
    period = 2 * np.pi * np.sqrt(radius**3 / (G * M_EARTH))
    return period

def analyical_circular_trajectory(radius, times):
    """
    Calculate the analytical trajectory of a circular orbit at a given radius.

    Parameters:
    radius (float): The radius of the circular orbit in meters.
    times (np.ndarray): The times at which to calculate the positions in seconds.

    Returns:
    np.ndarray: The position vectors of the object in the circular orbit at the given times.
    """
    angular_velocity = np.sqrt(
        G * M_EARTH
        / radius**3
    )

    angles = (
        angular_velocity
        * times
    )

    positions = np.column_stack(
        (
            radius * np.cos(angles),

            radius * np.sin(angles),

            np.zeros_like(times)
        )
    )

    velocities = np.column_stack(
        (
            -radius
            * angular_velocity
            * np.sin(angles),

            radius
            * angular_velocity
            * np.cos(angles),

            np.zeros_like(times)
        )
    )

    return positions, velocities