"""Reference-style figures for experiment 001, one per integrator.

The default run saves four 10,000 s figures and four 200,000 s (20x) figures.
Reuses simulation.py's initial-condition and propagation functions unchanged.
Uses the reference plot's 16x14 size, Matplotlib default fonts, four-panel
layout, labels, grid, line widths, legends and 300 dpi. Only method palettes
are customized. Keeps the reference overall title; no inset, subtitles or explanatory banners.
Each integrator's diagnostic panels autoscale independently.

Run: uv run python experiments/001_circular_orbit_convergence/plot_per_integrator.py
Options: --durations 10000 100000 (10x instead) and --show (interactive windows).
Output paths are resolved from this file, independent of the current directory.
"""
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from simulation import calculate_cicle_orbit_factor, simulate_cicle_orbit
from orbit_sim.dynamics import G, M_EARTH

DT_VALUES = (40, 20, 10, 5)
DURATION = 10000.0
DEFAULT_DURATIONS = (DURATION, 20 * DURATION)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "research" / "img"
METHODS = {
    "euler": {
        "title": "Euler",
        "colors": ("#0B3B8C", "#2463B4", "#438DD1", "#74B2E0"),
    },
    "symplectic": {
        "title": "Symplectic Euler",
        "colors": ("#8F2D0B", "#BF4B17", "#E27824", "#C69A1A"),
    },
    "velocity_verlet_KDK": {
        "title": "Velocity Verlet KDK",
        "colors": ("#105C3A", "#24864F", "#45AA68", "#78B87A"),
    },
    "velocity_verlet_DKD": {
        "title": "Velocity Verlet DKD",
        "colors": ("#52206D", "#793BA0", "#A15FC2", "#C58DCB"),
    },
}
LINE_STYLES = ("-", "--", "-.", ":")


def collect_results(duration=DURATION):
    """Keep the reference initial state and dt values; vary only end time."""
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    position = np.array([1.0e7, 0.0, 0.0])
    velocity, radius = calculate_cicle_orbit_factor(
        position=position, velocity_direction=np.array([0.0, 1.0, 0.0])
    )
    results = {}
    for dt in DT_VALUES:
        steps = round(duration / dt)
        if not np.isclose(steps * dt, duration, rtol=0, atol=1e-9):
            raise ValueError("duration must be divisible by every dt value")
        results[dt] = simulate_cicle_orbit(position, velocity, dt, steps)
    return results, radius


def diagnostic_series(result, radius, dt):
    """Preserve the analytical reference and error definitions of simulation.py."""
    positions = result["positions"]
    times = np.arange(len(positions)) * dt
    angles = np.sqrt(G * M_EARTH / radius**3) * times
    exact_positions = np.column_stack(
        (radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(times))
    )
    energy = result["energy"]
    angular = result["angular_momentum"]
    return {
        "times": times,
        "position_error": np.linalg.norm(positions - exact_positions, axis=1),
        "relative_energy_error": (energy - energy[0]) / abs(energy[0]),
        "relative_angular_error": np.linalg.norm(angular - angular[0], axis=1)
        / np.linalg.norm(angular[0]),
        "radius_deviation": np.linalg.norm(positions, axis=1) - radius,
    }


def output_name(method, duration):
    if np.isclose(duration, DURATION, rtol=0, atol=1e-9):
        return f"001_{method}_dt_comparison.png"
    factor = f"{duration / DURATION:g}".replace(".", "p")
    return f"001_{method}_dt_comparison_long_{factor}x.png"


def plot_method(all_data, radius, method, output_dir=OUTPUT_DIR):
    """Match simulation.py's four-panel format and overall title."""
    settings = METHODS[method]
    diagnostics = [diagnostic_series(all_data[dt][method], radius, dt) for dt in DT_VALUES]
    duration = float(diagnostics[0]["times"][-1])
    if not all(np.isclose(s["times"][-1], duration) for s in diagnostics):
        raise ValueError("Every dt curve must have the same end time")
    if not all(np.all(np.isfinite(value)) for series in diagnostics for value in series.values()):
        raise ValueError("Nonfinite diagnostic data")

    # Reset only plotting style so fonts, ticks, spines and grid match the
    # reference script's default Matplotlib settings, regardless of prior calls.
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        orbit_ax, position_ax, energy_ax, angular_ax = axes.flat
        theta = np.linspace(0.0, 2.0 * np.pi, 1000)
        orbit_ax.plot(radius * np.cos(theta), radius * np.sin(theta),
                      color="black", linewidth=2.5, label="Analytical orbit")
        orbit_ax.scatter(0.0, 0.0, color="black", s=60, marker="o")

        for dt, color, linestyle, series in zip(
            DT_VALUES, settings["colors"], LINE_STYLES, diagnostics
        ):
            positions = all_data[dt][method]["positions"]
            options = {"color": color, "linestyle": linestyle, "linewidth": 1.5}
            orbit_ax.plot(positions[:, 0], positions[:, 1], **options)
            position_ax.plot(series["times"], series["position_error"], **options)
            energy_ax.plot(series["times"], series["relative_energy_error"], **options)
            angular_ax.plot(series["times"], series["relative_angular_error"], **options)

        orbit_ax.set_title("Orbit Comparison")
        orbit_ax.set_xlabel("x (m)")
        orbit_ax.set_ylabel("y (m)")
        orbit_ax.axis("equal")
        position_ax.set_title("Position Error vs Time")
        position_ax.set_xlabel("Time (s)")
        position_ax.set_ylabel("Position Error (m)")
        energy_ax.set_title("Relative Energy Error vs Time")
        energy_ax.set_xlabel("Time (s)")
        energy_ax.set_ylabel("(E - E0) / |E0|")
        angular_ax.set_title("Relative Angular Momentum Error vs Time")
        angular_ax.set_xlabel("Time (s)")
        angular_ax.set_ylabel("|h - h0| / |h0|")

        method_handles = [Line2D([0], [0], color=settings["colors"][0],
                                 linewidth=2, label=settings["title"])]
        dt_handles = [Line2D([0], [0], color=color, linestyle=linestyle,
                             linewidth=2, label=f"dt = {dt} s")
                      for dt, color, linestyle in zip(DT_VALUES, settings["colors"], LINE_STYLES)]
        for ax in axes.flat:
            ax.grid()
            method_legend = ax.legend(handles=method_handles, title="Integrator",
                                      loc="upper left", fontsize=8)
            ax.add_artist(method_legend)
            ax.legend(handles=dt_handles, title="Time Step", loc="upper right", fontsize=8)

        # Default autoscaling is applied separately to each integrator's figure;
        # Euler drift therefore cannot flatten the other methods' oscillations.
        fig.suptitle("Numerical Orbit Integration Comparison", fontsize=16)
        fig.tight_layout()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / output_name(method, duration)
        fig.savefig(path, dpi=300)
        return fig, path


def main(show=False, durations=DEFAULT_DURATIONS):
    for duration in durations:
        all_data, radius = collect_results(duration)
        for method in METHODS:
            fig, path = plot_method(all_data, radius, method)
            print(f"Saved: {path}")
            if not show:
                plt.close(fig)
    if show:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Also show figures interactively")
    parser.add_argument("--durations", nargs="+", type=float, default=DEFAULT_DURATIONS,
                        help="Durations in seconds; default: 10000 200000")
    args = parser.parse_args()
    main(show=args.show, durations=args.durations)
