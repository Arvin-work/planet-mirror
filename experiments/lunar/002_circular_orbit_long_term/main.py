import math
from orbit_sim.dynamics import gravitational_acceleration, G, M_EARTH
from orbit_sim.integrators import eular, Symplectic_Euler, Velocity_Verlet_KDK, Velocity_Verlet_DKD
from orbit_sim.diagnostics import specific_energy, specific_angular_momentum
from orbit_sim.simulate import simulate_orbit_with_multiple_integrators
from orbit_sim.circular_orbit import (
    circular_orbit_velocity,
    circular_orbit_period,
    analyical_circular_trajectory,
)

import numpy as np
import matplotlib.pyplot as plt

def plot_long_term_results(
    data,
    exact_positions,
    times,
    radius,
    dt,
    orbit_count
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 13)
    )

    ax_orbit = axes[0, 0]
    ax_position_error = axes[0, 1]
    ax_energy = axes[1, 0]
    ax_angular = axes[1, 1]


    # =========================================================
    # Analytical orbit
    # =========================================================

    theta = np.linspace(
        0,
        2 * np.pi,
        1000
    )

    ax_orbit.plot(
        radius * np.cos(theta),
        radius * np.sin(theta),
        linestyle="--",
        color="black",
        label="Analytical orbit"
    )

    ax_orbit.scatter(
        0.0,
        0.0,
        color="black",
        s=50,
        label="Earth"
    )


    # =========================================================
    # Numerical methods
    # =========================================================

    for method_name, result in data.items():

        positions = result["positions"]
        energy = result["energy"]
        velocity = result["velocities"]
        angular_momentum = result[
            "anguluar_momentum"
        ]


        # -----------------------------------------------------
        # Orbit
        # -----------------------------------------------------

        ax_orbit.plot(
            positions[:, 0],
            positions[:, 1],
            label=method_name
        )


        # -----------------------------------------------------
        # Position error
        # -----------------------------------------------------

        position_error = np.linalg.norm(
            positions - exact_positions,
            axis=1
        )

        ax_position_error.plot(
            times,
            position_error,
            label=method_name
        )


        # -----------------------------------------------------
        # Relative energy error
        # -----------------------------------------------------

        relative_energy_error = (
            (energy - energy[0])
            / abs(energy[0])
        )

        ax_energy.plot(
            times,
            relative_energy_error,
            label=method_name
        )


        # -----------------------------------------------------
        # Relative angular momentum error
        # -----------------------------------------------------

        angular_error = np.linalg.norm(
            angular_momentum
            - angular_momentum[0],
            axis=1
        )

        relative_angular_error = (
            angular_error
            / np.linalg.norm(
                angular_momentum[0]
            )
        )

        ax_angular.plot(
            times,
            relative_angular_error,
            label=method_name
        )


        # -----------------------------------------------------
        # Numerical summary
        # -----------------------------------------------------

        radius_values = np.linalg.norm(
            positions,
            axis=1
        )

        print(
            f"\n--- {method_name} ---"
        )

        print(
            "Orbit radius min:",
            np.min(radius_values)
        )

        print(
            "Orbit radius max:",
            np.max(radius_values)
        )

        print(
            "Position error max:",
            np.max(position_error)
        )

        print(
            "Relative energy error min:",
            np.min(relative_energy_error)
        )

        print(
            "Relative energy error max:",
            np.max(relative_energy_error)
        )

        print(
            "Relative angular momentum error max:",
            np.max(relative_angular_error)
        )


    # =========================================================
    # Figure settings
    # =========================================================

    ax_orbit.set_title(
        "Long-term Orbit"
    )

    ax_orbit.set_xlabel("x (m)")
    ax_orbit.set_ylabel("y (m)")
    ax_orbit.axis("equal")
    ax_orbit.grid()
    ax_orbit.legend()


    ax_position_error.set_title(
        "Position Error vs Time"
    )

    ax_position_error.set_xlabel(
        "Time (s)"
    )

    ax_position_error.set_ylabel(
        "Position Error (m)"
    )

    ax_position_error.grid()
    ax_position_error.legend()


    ax_energy.set_title(
        "Relative Energy Error vs Time"
    )

    ax_energy.set_xlabel(
        "Time (s)"
    )

    ax_energy.set_ylabel(
        "(E - E0) / |E0|"
    )

    ax_energy.grid()
    ax_energy.legend()


    ax_angular.set_title(
        "Relative Angular Momentum Error vs Time"
    )

    ax_angular.set_xlabel(
        "Time (s)"
    )

    ax_angular.set_ylabel(
        "|h - h0| / |h0|"
    )

    ax_angular.grid()
    ax_angular.legend()


    fig.suptitle(
        f"Long-term Circular Orbit\n"
        f"dt = {dt} s, "
        f"target orbit count = {orbit_count}"
    )

    plt.tight_layout()

    plt.show()

if __name__ == "__main__":

    # =========================================================
    # 1. Initial condition
    # =========================================================

    initial_position = np.array([
        1.0e7,
        0.0,
        0.0
    ])

    velocity_direction = np.array([
        0.0,
        1.0,
        0.0
    ])

    radius = np.linalg.norm(
        initial_position
    )

    initial_velocity = circular_orbit_velocity(
        position=initial_position,
        velocity_direction=velocity_direction
    )

    print(
    "Initial position:",
    initial_position
    )

    print(
        "Initial velocity:",
        initial_velocity
    )

    print(
        "Initial velocity norm:",
        np.linalg.norm(initial_velocity)
    )

    print(
        "Initial angular momentum:",
        specific_angular_momentum(
            position=initial_position,
            velocity=initial_velocity
        )
    )
    
    integrators = {
        "Euler": eular,
        "Symplectic Euler": Symplectic_Euler,
        "Velocity Verlet KDK": Velocity_Verlet_KDK,
        "Velocity Verlet DKD": Velocity_Verlet_DKD
    }

    # =========================================================
    # 2. Long-term experiment settings
    # =========================================================

    dt = 20.0
    orbit_count = 20
    period = circular_orbit_period(radius=radius)
    total_time = (orbit_count * period)
    steps = int(total_time / dt)
    actual_time = (steps * dt)


    print("dt:",dt,"s")
    print("Orbit count:",orbit_count)
    print("Simulation time:",actual_time,"s")
    print("Steps:",steps)

    # =========================================================
    # 3. Run simulation
    # =========================================================

    data = simulate_orbit_with_multiple_integrators(
        position=initial_position,
        velocity=initial_velocity,
        dt=dt,
        steps=steps,
        integrators=integrators
    )

    # =========================================================
    # 4. Time axis
    # =========================================================

    times = (
        np.arange(steps + 1)
        * dt
    )

    exact_positions, exact_velocities = (
        analyical_circular_trajectory(
            radius=radius,
            times=times
        )
    )

    plot_long_term_results(
        data=data,
        exact_positions=exact_positions,
        times=times,
        radius=radius,
        dt=dt,
        orbit_count=orbit_count
    )