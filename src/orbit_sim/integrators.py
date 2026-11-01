"""Position-only integrators for (3,) or (N,3) states.

Optional acceleration_function(position) must return the same shape. Omitting
it preserves the fixed-origin Earth model (also independently per array row).
"""
from . import dynamics
import numpy as np

def eular(position, velocity, dt, acceleration_function=None):
    """
    Perform one step of the Euler integration method.

    Parameters:
    position (numpy.ndarray): Current position vector.
    velocity (numpy.ndarray): Current velocity vector.
    dt (float): Time step for integration.

    This Program will use dynamic to calculate 
    acceleration (numpy.ndarray): Current acceleration vector.
    
    Returns:
    tuple: Updated position and velocity vectors after one time step.
    """
    position, velocity, force = _prepare(position, velocity, dt, acceleration_function)
    acceleration = force(position)

    new_position = position + velocity * dt
    new_velocity = velocity + acceleration * dt
    return new_position, new_velocity

def Symplectic_Euler(position, velocity, dt, acceleration_function=None):
    """
    Perform one step of the Symplectic Euler integration method.

    Parameters:
    position (numpy.ndarray): Current position vection;
    velocity (numpy.ndarray): Current velocity vection;
    dt (float): Time step for integration 

    Returns:
    tuple: Update the position and velocity vectors after one time step.
    """

    position, velocity, force = _prepare(position, velocity, dt, acceleration_function)
    acceleration = force(position)

    velocity_new = velocity + acceleration * dt
    position_new = position + velocity_new * dt

    return position_new, velocity_new

def Velocity_Verlet_KDK(position, velocity, dt, acceleration_function=None):
    """
    Perform one step of the Velocity Verlet integration method.

    Parameters:
    position (numpy.ndarray): Current position vector.
    velocity (numpy.ndarray): Current velocity vector.
    dt (float): Time step for integration.

    Returns:
    tuple: Updated position and velocity vectors after one time step.
    """
    
    position, velocity, force = _prepare(position, velocity, dt, acceleration_function)
    acceleration = force(position)

    position_new = position + velocity * dt + 0.5 * acceleration * dt**2
    acceleration_new = force(position_new)
    velocity_new = velocity + 0.5 * (acceleration + acceleration_new) *dt
    return position_new, velocity_new

def Velocity_Verlet_DKD(position, velocity, dt, acceleration_function=None):
    """
    Perform one step of the Velocity Verlet integration method using the DKD scheme.

    Parameters:
    position (numpy.ndarray): Current position vector.
    velocity (numpy.ndarray): Current velocity vector.
    dt (float): Time step for integration.

    Returns:
    tuple: Updated position and velocity vectors after one time step.
    """
    
    position, velocity, force = _prepare(position, velocity, dt, acceleration_function)
    position_half = position + 0.5 * velocity * dt
    acceleration_half = force(position_half)
    velocity_new = velocity + acceleration_half * dt
    position_new = position_half + 0.5 * velocity_new * dt

    return position_new, velocity_new


def _prepare(position, velocity, dt, acceleration_function):
    """Force callbacks are pure position-only functions, called on full snapshots."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if position.shape != velocity.shape or not (
        position.shape == (3,) or (position.ndim == 2 and position.shape[1] == 3)
    ):
        raise ValueError("position/velocity must have matching (3,) or (N,3) shapes")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)) or not np.isfinite(dt):
        raise ValueError("state and dt must be finite")
    callback = acceleration_function
    if callback is None:
        def callback(r):
            if r.ndim == 1:
                return dynamics.gravitational_acceleration(r)
            return np.array([dynamics.gravitational_acceleration(row) for row in r]).reshape(r.shape)

    def force(r):
        snapshot = r.copy()
        snapshot.setflags(write=False)
        result = np.asarray(callback(snapshot), dtype=float)
        if result.shape != r.shape or not np.all(np.isfinite(result)):
            raise ValueError("acceleration function must return finite acceleration matching state shape")
        return result

    return position, velocity, force


# Preserve historical names while offering a correctly spelled alias.
euler = eular
