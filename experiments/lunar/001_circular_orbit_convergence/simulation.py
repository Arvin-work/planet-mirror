import numpy as np
import math
import matplotlib.pyplot as plt
from orbit_sim.dynamics import gravitational_acceleration, G, M_EARTH
from orbit_sim.integrators import eular, Symplectic_Euler, Velocity_Verlet_KDK, Velocity_Verlet_DKD
from orbit_sim.diagnostics import specific_energy, specific_angular_momentum

def calculate_cicle_orbit_factor(position: np.ndarray, velocity_direction: np.ndarray):
    """
    Calculate the orbital factor for a circular orbit based on the given position and velocity.
    It returns the radio, value of velocity, velocity.
    """
    direction_norm = np.linalg.norm(velocity_direction)

    if direction_norm == 0:
        raise ValueError("Velocity direction cannot be zero.")

    velocity_direction = velocity_direction / direction_norm

    radius = np.linalg.norm(position)
    value_velocity = math.sqrt((G * M_EARTH) / radius)
    velocity = value_velocity * velocity_direction

    return velocity, radius



def simulate_cicle_orbit(position, velocity, dt, steps):
    """
    
    """

    first_position = position.copy()
    first_velocity = velocity.copy()

    euler_positions = [first_position.copy()]
    euler_velocity = [first_velocity.copy()]
    euler_energy = [specific_energy(position=first_position, velocity=first_velocity)]
    euler_angular_momentum = [specific_angular_momentum(position=first_position, velocity=first_velocity)]

    symplectic_positions = [first_position.copy()]
    symplectic_velocity = [first_velocity.copy()]
    symplectic_energy = [specific_energy(position=first_position, velocity=first_velocity)]
    symplectic_angular_momentum = [specific_angular_momentum(position=first_position, velocity=first_velocity)]

    velocity_verlet_KDK_positions = [first_position.copy()]
    velocity_verlet_KDK_velocity = [first_velocity.copy()]
    velocity_verlet_KDK_energy = [specific_energy(position=first_position, velocity=first_velocity)]
    velocity_verlet_KDK_angular_momentum = [specific_angular_momentum(position=first_position, velocity=first_velocity)]

    velocity_verlet_DKD_positions = [first_position.copy()]
    velocity_verlet_DKD_velocity = [first_velocity.copy()]
    velocity_verlet_DKD_energy = [specific_energy(position=first_position, velocity=first_velocity)]
    velocity_verlet_DKD_angular_momentum = [specific_angular_momentum(position=first_position, velocity=first_velocity)]

    for _ in range(steps):
        new_position, new_velocity = eular(position=position, velocity=velocity, dt=dt)
        euler_energy_value = specific_energy(position=new_position, velocity=new_velocity)
        euler_angular_momentum_value = specific_angular_momentum(position=new_position, velocity=new_velocity)
        euler_positions.append(new_position)
        euler_velocity.append(new_velocity)
        euler_energy.append(euler_energy_value)
        euler_angular_momentum.append(euler_angular_momentum_value)
        
        position, velocity = new_position, new_velocity

    position, velocity = first_position.copy(), first_velocity.copy()
    for _ in range(steps):
        new_position, new_velocity = Symplectic_Euler(position=position, velocity=velocity, dt=dt)
        symplectic_energy_value = specific_energy(position=new_position, velocity=new_velocity)
        symplectic_angular_momentum_value = specific_angular_momentum(position=new_position, velocity=new_velocity)
        symplectic_positions.append(new_position)
        symplectic_velocity.append(new_velocity)
        symplectic_energy.append(symplectic_energy_value)
        symplectic_angular_momentum.append(symplectic_angular_momentum_value)

        position, velocity = new_position, new_velocity

    position, velocity = first_position.copy(), first_velocity.copy()
    for _ in range(steps):
        new_position, new_velocity = Velocity_Verlet_KDK(position=position, velocity=velocity, dt=dt)
        velocity_verlet_KDK_energy_value = specific_energy(position=new_position, velocity=new_velocity)
        velocity_verlet_KDK_angular_momentum_value = specific_angular_momentum(position=new_position, velocity=new_velocity)
        velocity_verlet_KDK_positions.append(new_position)
        velocity_verlet_KDK_velocity.append(new_velocity)
        velocity_verlet_KDK_energy.append(velocity_verlet_KDK_energy_value)
        velocity_verlet_KDK_angular_momentum.append(velocity_verlet_KDK_angular_momentum_value)

        position, velocity = new_position, new_velocity

    position, velocity = first_position.copy(), first_velocity.copy()
    for _ in range(steps):
        new_position, new_velocity = Velocity_Verlet_DKD(position=position, velocity=velocity, dt=dt)
        velocity_verlet_DKD_energy_value = specific_energy(position=new_position, velocity=new_velocity)
        velocity_verlet_DKD_angular_momentum_value = specific_angular_momentum(position=new_position, velocity=new_velocity)
        velocity_verlet_DKD_positions.append(new_position)
        velocity_verlet_DKD_velocity.append(new_velocity)
        velocity_verlet_DKD_energy.append(velocity_verlet_DKD_energy_value)
        velocity_verlet_DKD_angular_momentum.append(velocity_verlet_DKD_angular_momentum_value)

        position, velocity = new_position, new_velocity

    return {
        "euler": {
            "positions": np.array(euler_positions),
            "velocity": np.array(euler_velocity),
            "energy": np.array(euler_energy),
            "angular_momentum": np.array(euler_angular_momentum)
        },
        "symplectic": {
            "positions": np.array(symplectic_positions),
            "velocity": np.array(symplectic_velocity),
            "energy": np.array(symplectic_energy),
            "angular_momentum": np.array(symplectic_angular_momentum)
        },
        "velocity_verlet_KDK": {
            "positions": np.array(velocity_verlet_KDK_positions),
            "velocity": np.array(velocity_verlet_KDK_velocity),
            "energy": np.array(velocity_verlet_KDK_energy),
            "angular_momentum": np.array(velocity_verlet_KDK_angular_momentum)
        },
        "velocity_verlet_DKD": {
            "positions": np.array(velocity_verlet_DKD_positions),
            "velocity": np.array(velocity_verlet_DKD_velocity),
            "energy": np.array(velocity_verlet_DKD_energy),
            "angular_momentum": np.array(velocity_verlet_DKD_angular_momentum)
        }
    }

def plot_all_simulation_results(all_data, radius):
    """
    Plot all dt values and all integrators in ONE figure.

    Subplot 1:
        All numerical orbits + analytical circular orbit

    Subplot 2:
        Position error vs time

    Subplot 3:
        Relative energy error vs time

    Subplot 4:
        Relative angular momentum error vs time

    Parameters
    ----------
    all_data : dict
        Structure:

        {
            40: data_dt40,
            20: data_dt20,
            10: data_dt10,
            5: data_dt5
        }

        Every data_dt contains:
        {
            "euler": {...},
            "symplectic": {...},
            "velocity_verlet_KDK": {...},
            "velocity_verlet_DKD": {...}
        }

    radius : float
        Analytical circular orbit radius.
    """

    from matplotlib.lines import Line2D


    # =========================================================
    # Figure
    # =========================================================

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 14)
    )

    ax_orbit = axes[0, 0]
    ax_position = axes[0, 1]
    ax_energy = axes[1, 0]
    ax_angular = axes[1, 1]


    # =========================================================
    # Use color to represent integrator
    # Use line style to represent dt
    # =========================================================

    method_colors = {
        "euler": "tab:blue",
        "symplectic": "tab:orange",
        "velocity_verlet_KDK": "tab:green",
        "velocity_verlet_DKD": "tab:red"
    }

    dt_styles = {
        40: "-",
        20: "--",
        10: "-.",
        5: ":"
    }


    # =========================================================
    # Analytical orbit
    # =========================================================

    theta = np.linspace(
        0,
        2 * np.pi,
        1000
    )

    exact_circle_x = (
        radius * np.cos(theta)
    )

    exact_circle_y = (
        radius * np.sin(theta)
    )

    ax_orbit.plot(
        exact_circle_x,
        exact_circle_y,
        color="black",
        linewidth=2.5,
        label="Analytical orbit"
    )

    ax_orbit.scatter(
        0,
        0,
        color="black",
        s=60,
        marker="o"
    )


    # =========================================================
    # Loop through ALL dt
    # =========================================================

    for dt, data in all_data.items():

        print(
            "\n"
            "============================================"
        )

        print(
            f"dt = {dt} s"
        )

        print(
            "============================================"
        )


        # =====================================================
        # ALL four integrators
        # =====================================================

        for method_name, result in data.items():

            positions = result["positions"]

            energy = result["energy"]

            angular_momentum = (
                result["angular_momentum"]
            )


            # =================================================
            # Time
            # =================================================

            steps = len(positions) - 1

            times = (
                np.arange(steps + 1)
                * dt
            )


            # =================================================
            # Analytical position for every time
            # =================================================

            angular_velocity = math.sqrt(
                G * M_EARTH
                / radius**3
            )

            angle = (
                angular_velocity
                * times
            )

            exact_positions = np.column_stack(
                (
                    radius * np.cos(angle),

                    radius * np.sin(angle),

                    np.zeros_like(times)
                )
            )


            # =================================================
            # 1. Orbit
            # =================================================

            ax_orbit.plot(
                positions[:, 0],
                positions[:, 1],
                color=method_colors[
                    method_name
                ],
                linestyle=dt_styles[dt],
                linewidth=1.5
            )


            # =================================================
            # 2. Position error
            # =================================================

            position_error = np.linalg.norm(
                positions
                - exact_positions,
                axis=1
            )

            ax_position.plot(
                times,
                position_error,
                color=method_colors[
                    method_name
                ],
                linestyle=dt_styles[dt],
                linewidth=1.5
            )


            # =================================================
            # 3. Relative energy error
            # =================================================

            initial_energy = energy[0]

            relative_energy_error = (
                (energy - initial_energy)
                / abs(initial_energy)
            )

            ax_energy.plot(
                times,
                relative_energy_error,
                color=method_colors[
                    method_name
                ],
                linestyle=dt_styles[dt],
                linewidth=1.5
            )


            # =================================================
            # 4. Relative angular momentum error
            # =================================================

            initial_angular_momentum = (
                angular_momentum[0]
            )

            angular_error = np.linalg.norm(
                angular_momentum
                - initial_angular_momentum,
                axis=1
            )

            relative_angular_error = (
                angular_error
                / np.linalg.norm(
                    initial_angular_momentum
                )
            )

            ax_angular.plot(
                times,
                relative_angular_error,
                color=method_colors[
                    method_name
                ],
                linestyle=dt_styles[dt],
                linewidth=1.5
            )


            # =================================================
            # Radius
            # =================================================

            radius_values = np.linalg.norm(
                positions,
                axis=1
            )


            # =================================================
            # Print Min / Max
            # =================================================

            print(
                f"\n--- {method_name} ---"
            )

            print(
                "Orbit Radius:"
            )

            print(
                "    Min:",
                np.min(radius_values),
                "m"
            )

            print(
                "    Max:",
                np.max(radius_values),
                "m"
            )


            print(
                "Position Error:"
            )

            print(
                "    Min:",
                np.min(position_error),
                "m"
            )

            print(
                "    Max:",
                np.max(position_error),
                "m"
            )


            print(
                "Relative Energy Error:"
            )

            print(
                "    Min:",
                np.min(
                    relative_energy_error
                )
            )

            print(
                "    Max:",
                np.max(
                    relative_energy_error
                )
            )


            print(
                "Relative Angular Momentum Error:"
            )

            print(
                "    Min:",
                np.min(
                    relative_angular_error
                )
            )

            print(
                "    Max:",
                np.max(
                    relative_angular_error
                )
            )


    # =========================================================
    # Orbit subplot
    # =========================================================

    ax_orbit.set_title(
        "Orbit Comparison"
    )

    ax_orbit.set_xlabel(
        "x (m)"
    )

    ax_orbit.set_ylabel(
        "y (m)"
    )

    ax_orbit.axis(
        "equal"
    )

    ax_orbit.grid()


    # =========================================================
    # Position error subplot
    # =========================================================

    ax_position.set_title(
        "Position Error vs Time"
    )

    ax_position.set_xlabel(
        "Time (s)"
    )

    ax_position.set_ylabel(
        "Position Error (m)"
    )

    ax_position.grid()


    # =========================================================
    # Energy subplot
    # =========================================================

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


    # =========================================================
    # Angular momentum subplot
    # =========================================================

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


    # =========================================================
    # Legend 1:
    # Colors = integrators
    # =========================================================

    method_legend = []

    for method_name, color in (
        method_colors.items()
    ):

        method_legend.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=2,
                label=method_name
            )
        )


    # =========================================================
    # Legend 2:
    # Line styles = dt
    # =========================================================

    dt_legend = []

    for dt, linestyle in (
        dt_styles.items()
    ):

        dt_legend.append(
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=linestyle,
                linewidth=2,
                label=f"dt = {dt} s"
            )
        )


    # =========================================================
    # Put legends on every subplot
    # =========================================================

    for ax in axes.flatten():

        legend_methods = ax.legend(
            handles=method_legend,
            title="Integrator",
            loc="upper left",
            fontsize=8
        )

        ax.add_artist(
            legend_methods
        )

        ax.legend(
            handles=dt_legend,
            title="Time Step",
            loc="upper right",
            fontsize=8
        )


    # =========================================================
    # Whole figure
    # =========================================================

    fig.suptitle(
        "Numerical Orbit Integration Comparison",
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(fname="research/img/001_orbit_integration_comparison.png", dpi=300,)
    plt.show()

def analytial_circular_state(radius, time):
    """
    Calculate the analytical position and velocity for a circular orbit at a given time.

    Parameters:
    radius (float): Radius of the circular orbit.
    time (float): Time at which to calculate the state.

    Returns:
    tuple: Analytical position and velocity vectors.
    """
    angular_velocity = math.sqrt(G * M_EARTH / radius**3)
    angle = angular_velocity * time

    position = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])
    velocity = np.array([-radius * angular_velocity * np.sin(angle), radius * angular_velocity * np.cos(angle), 0.0])

    return position, velocity
    
if __name__ == '__main__':
    all_data = {}
    position = np.array([1.0e7, 0.0, 0.0])
    velocity_direction = np.array([0.0, 1.0, 0.0])

    direction_norm = np.linalg.norm(velocity_direction)
    velocity_direction = velocity_direction / direction_norm

    dt_value = [40, 20, 10, 5]


    velocity, radius = calculate_cicle_orbit_factor(position=position, velocity_direction=velocity_direction)
    initial_energy = specific_energy(position=position, velocity=velocity)
    initial_angular_momentum = specific_angular_momentum(position=position, velocity=velocity)
    print(f"Calculated circular orbit velocity: {velocity}, radius: {radius}")

    error_data = {"position_error": [], "velocity_error": [], "energy_error": [], "angular_momentum_error": []}

    for dt in dt_value:
        steps = int(10000 / dt)
        print("--- simulation factor information ---")
        print("dt:", dt)
        print("steps:", steps)
        data = simulate_cicle_orbit(position=position, velocity=velocity, dt=dt, steps=steps)
        all_data[dt] = data
        expected_data = analytial_circular_state(radius=radius, time=dt * steps)

        for method_name, result in data.items():
            print(f"\n--- {method_name} ---")

            print(
                "Final position:",
                result["positions"][-1]
            )

            print(
                "Final velocity:",
                result["velocity"][-1]
            )

            print(
                "Final energy:",
                result["energy"][-1]
            )

            print(
                "Final angular momentum:",
                result["angular_momentum"][-1]
            )

            energy_error = (
                result["energy"][-1]
                - initial_energy
            )
            error_data["energy_error"].append(energy_error)

            angular_momentum_error = (
                result["angular_momentum"][-1] 
                - initial_angular_momentum
            )
            error_data["angular_momentum_error"].append(angular_momentum_error)

            print("Energy Error:", energy_error)
            print("Angular Error:", angular_momentum_error)

            position_error = (
                result["positions"][-1]
                - expected_data[0]
            )
            position_error = np.linalg.norm(position_error)
            error_data["position_error"].append(position_error)

            print("Position Error:", position_error)

            velocity_error = (
                result["velocity"][-1]
                - expected_data[1]
            )
            velocity_error = np.linalg.norm(velocity_error)
            error_data["velocity_error"].append(velocity_error)

            print("Velocity Error:", velocity_error)

    print("\n--- Error Summary ---")
    name = ['Euler', 'Symplectic Euler', 'Velocity Verlet KDK', 'Velocity Verlet DKD']
    for index in range(4):
        print(f"\n--- {name[index]} ---")
        for i in range(3):
            old = dt_value[i]
            new = dt_value[i + 1]
            energy_error_old = error_data["energy_error"][i * 4 + index]
            energy_error_new = error_data["energy_error"][i * 4 + index + 4]
            order = math.log(abs(energy_error_old)/abs(energy_error_new), 2)

            print(f"{old} -> {new}: p =", order)

    plot_all_simulation_results(
    all_data=all_data,
    radius=radius)