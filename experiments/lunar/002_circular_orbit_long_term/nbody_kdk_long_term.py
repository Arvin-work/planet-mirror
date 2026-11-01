"""Earth/Moon + passive spacecraft: a separate long-term KDK experiment.

Run from the project root:
    .venv/bin/python experiments/002_circular_orbit_long_term/nbody_kdk_long_term.py

The old main.py is unchanged. Units: m, s, kg; inertial barycentric frame.
The two long runs end at exactly the same time. The final spacecraft phase
may differ substantially even when both trajectories remain bounded.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from orbit_sim.dynamics import G, M_EARTH, nbody_accelerations
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.simulate import simulate_nbody

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "research" / "img"
EARTH, MOON, SPACECRAFT = 0, 1, 2
EARTH_RADIUS = 6.371e6
SPACECRAFT_RADIUS = 1.e7  # Distance from Earth's center, not altitude.
EARTH_MOON_DISTANCE = 3.844e8


def initial_state():
    """Create the four arrays. Row order: Earth, Moon, spacecraft."""
    masses = np.array([M_EARTH, 7.342e22, 1000.])  # (3,), kg
    gravitating = np.array([True, True, False])   # (3,), source mask

    # Ideal circular Earth/Moon pair, with its center of mass at the origin.
    earth_fraction, moon_fraction = masses[:2] / masses[:2].sum()
    relative_speed = np.sqrt(G * masses[:2].sum() / EARTH_MOON_DISTANCE)
    earth_x = -moon_fraction * EARTH_MOON_DISTANCE
    moon_x = earth_fraction * EARTH_MOON_DISTANCE
    earth_vy = -moon_fraction * relative_speed
    moon_vy = earth_fraction * relative_speed
    local_spacecraft_speed = np.sqrt(G * masses[EARTH] / SPACECRAFT_RADIUS)

    positions = np.array([                      # (3, 3), m
        [earth_x, 0., 0.],
        [moon_x, 0., 0.],
        [earth_x + SPACECRAFT_RADIUS, 0., 0.],
    ])
    velocities = np.array([                     # (3, 3), m/s
        [0., earth_vy, 0.],
        [0., moon_vy, 0.],
        [0., earth_vy + local_spacecraft_speed, 0.],
    ])
    moon_period = 2 * np.pi * EARTH_MOON_DISTANCE / relative_speed
    spacecraft_period = 2 * np.pi * SPACECRAFT_RADIUS / local_spacecraft_speed
    return positions, velocities, masses, gravitating, moon_period, spacecraft_period


def propagate(positions, velocities, masses, gravitating, dt, steps):
    """One call propagates all three rows synchronously; no object updates."""
    return simulate_nbody(
        positions=positions, velocities=velocities, masses=masses,
        gravitating=gravitating, dt=dt, steps=steps,
        integrator=Velocity_Verlet_KDK,
    )


def check_passive_body(positions, velocities, masses, gravitating, dt):
    """A passive spacecraft must not change the active bodies' trajectory."""
    all_acceleration = nbody_accelerations(positions, masses, gravitating)
    active_acceleration = nbody_accelerations(positions[:2], masses[:2])
    np.testing.assert_array_equal(all_acceleration[:2], active_acceleration)
    all_bodies = propagate(positions, velocities, masses, gravitating, dt, 500)
    active_only = propagate(positions[:2], velocities[:2], masses[:2],
                            gravitating[:2], dt, 500)
    for key in ("positions", "velocities"):
        np.testing.assert_array_equal(all_bodies[key][:, :2], active_only[key])


def check_short_convergence(positions, velocities, masses, gravitating, dt, period):
    """Three resolutions at one aligned endpoint, about 5 spacecraft periods.

    There is no analytic spacecraft solution in the moving Earth/Moon field.
    Compare dt with dt/2, and dt/2 with dt/4; second order predicts a ratio ~4.
    This checks a short interval, not the full long-term spacecraft phase.
    """
    steps = max(1, round(5 * period / dt))
    endpoints = []
    for factor in (1, 2, 4):
        data = propagate(positions, velocities, masses, gravitating,
                         dt / factor, steps * factor)
        final = data["positions"][-1]
        endpoints.append(final[SPACECRAFT] - final[EARTH])
    differences = [np.linalg.norm(endpoints[i] - endpoints[i + 1]) for i in (0, 1)]
    ratio = differences[0] / differences[1]
    assert 3.7 < ratio < 4.3, f"Short-interval convergence ratio: {ratio}"
    return {"duration_s": steps * dt, "position_differences_m": differences,
            "ratio": float(ratio)}


def analyze(data, positions, velocities, masses, moon_period, spacecraft_period):
    """Verify full-resolution histories, then retain compact plotting data."""
    assert all(np.all(np.isfinite(array)) for array in data.values())
    time = data["times"]
    energy_error = np.abs((data["total_energy"] - data["total_energy"][0])
                          / abs(data["total_energy"][0]))
    angular_error = np.linalg.norm(data["total_angular_momentum"]
                                   - data["total_angular_momentum"][0], axis=1)
    angular_error /= np.linalg.norm(data["total_angular_momentum"][0])
    momentum_scale = np.sum(masses[:2] * np.linalg.norm(velocities[:2], axis=1))
    momentum_error = np.linalg.norm(data["total_linear_momentum"]
                                    - data["total_linear_momentum"][0], axis=1)
    momentum_error /= momentum_scale  # P0 is nearly zero; do not divide by P0.
    expected_com = (data["center_of_mass"][0]
                    + time[:, None] * data["total_linear_momentum"][0] / masses[:2].sum())
    com_error = np.linalg.norm(data["center_of_mass"] - expected_com, axis=1)

    relative_positions = data["positions"][:, SPACECRAFT] - data["positions"][:, EARTH]
    radius = np.linalg.norm(relative_positions, axis=1)
    extrema = {"energy": float(energy_error.max()),
               "angular_momentum": float(angular_error.max()),
               "linear_momentum": float(momentum_error.max()),
               "center_of_mass_drift_m": float(com_error.max()),
               "spacecraft_radius_min_m": float(radius.min()),
               "spacecraft_radius_max_m": float(radius.max())}
    assert extrema["energy"] < 1.e-6
    assert max(extrema["angular_momentum"], extrema["linear_momentum"]) < 1.e-10
    assert extrema["center_of_mass_drift_m"] / EARTH_MOON_DISTANCE < 1.e-10
    assert radius.min() > EARTH_RADIUS, "Spacecraft intersects Earth's surface"
    assert np.max(np.abs(radius / SPACECRAFT_RADIUS - 1)) < 0.1, "Large radial departure"

    # Plot one min/max envelope per approximate spacecraft orbit. Every sample
    # participates; this avoids aliasing short oscillations in a 550-day plot.
    dt = time[1] - time[0]
    block = max(1, round(spacecraft_period / dt))
    starts = np.arange(0, len(time), block)
    ends = np.minimum(starts + block - 1, len(time) - 1)
    envelope_time = (time[starts] + time[ends]) / (2 * moon_period)
    lower = np.minimum.reduceat(radius, starts) / 1000
    upper = np.maximum.reduceat(radius, starts) / 1000

    indices = np.unique(np.linspace(0, len(time) - 1, min(len(time), 20000), dtype=int))
    tail_steps = max(1, round(5 * spacecraft_period / dt))
    compact = {
        "positions": data["positions"][indices],
        "time_in_moon_periods": time[indices] / moon_period,
        "energy_error": energy_error[indices],
        "angular_error": angular_error[indices],
        "momentum_error": momentum_error[indices],
        "spacecraft_tail": relative_positions[-tail_steps - 1:],
        "final_relative_position": relative_positions[-1].copy(),
        "final_positions": data["positions"][-1].copy(),
        "envelope_time": envelope_time, "radius_min_km": lower, "radius_max_km": upper,
    }
    return compact, extrema


def plot_results(coarse, fine, dt, duration, moon_period, output_dir):
    """Four panels: two frames, active conservation, spacecraft radius."""
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        orbit_ax, local_ax, error_ax, radius_ax = axes.flat
        colors = ("#2463B4", "#C69A1A", "#24864F")
        for body, name, color in zip((EARTH, MOON, SPACECRAFT),
                                     ("Earth", "Moon", "Spacecraft"), colors):
            track = coarse["positions"][:, body] / 1000
            orbit_ax.plot(track[:, 0], track[:, 1], color=color, linewidth=1.2, label=name)
            final = coarse["final_positions"][body] / 1000
            orbit_ax.scatter(*final[:2], color=color, s=45, zorder=5)
        orbit_ax.scatter(0, 0, color="black", marker="+", label="Active center of mass")
        orbit_ax.set(title="Barycentric Trajectories (dots: final positions)",
                     xlabel="x (km)", ylabel="y (km)")
        orbit_ax.axis("equal")
        orbit_ax.legend(fontsize=9)

        for result, color, style, step in ((coarse, colors[2], "-", dt),
                                          (fine, "#793BA0", "--", dt / 2)):
            tail = result["spacecraft_tail"] / 1000
            local_ax.plot(tail[:, 0], tail[:, 1], color=color, linestyle=style,
                          linewidth=1.3, label=f"dt = {step:g} s")
            local_ax.scatter(*tail[-1, :2], color=color, s=45, zorder=5)
        local_ax.add_patch(plt.Circle((0, 0), EARTH_RADIUS / 1000, color=colors[0], alpha=.2))
        local_ax.set(title="Spacecraft Relative to Earth (last ~5 local periods)",
                     xlabel="Relative x (km)", ylabel="Relative y (km)")
        local_ax.axis("equal")
        local_ax.legend(fontsize=9)

        for key, label, color in (("energy_error", "Energy", colors[0]),
                                  ("angular_error", "Angular momentum", colors[2]),
                                  ("momentum_error", "Linear momentum", "#793BA0")):
            error_ax.semilogy(fine["time_in_moon_periods"], np.maximum(fine[key], 1.e-18),
                              color=color, linewidth=1.2, label=f"{label}, dt = {dt / 2:g} s")
        error_ax.semilogy(coarse["time_in_moon_periods"], np.maximum(coarse["energy_error"], 1.e-18),
                          color=colors[0], linestyle="--", label=f"Energy, dt = {dt:g} s")
        error_ax.set(title="Earth + Moon Conservation (active bodies only)",
                     xlabel="Time / initial Earth-Moon period", ylabel="Normalized absolute error")
        error_ax.legend(fontsize=8)

        for result, color, step in ((coarse, colors[2], dt), (fine, "#793BA0", dt / 2)):
            radius_ax.fill_between(result["envelope_time"], result["radius_min_km"],
                                   result["radius_max_km"], color=color, alpha=.25,
                                   label=f"Local-orbit min/max, dt = {step:g} s")
        radius_ax.axhline(SPACECRAFT_RADIUS / 1000, color="black", linestyle=":", label="Initial distance")
        radius_ax.set(title="Spacecraft Distance from Earth's Center",
                      xlabel="Time / initial Earth-Moon period", ylabel="Distance (km)")
        radius_ax.legend(fontsize=8)
        for ax in axes.flat:
            ax.grid()
        fig.suptitle(f"Earth-Moon-Spacecraft: Long-term KDK ({duration / 86400:.1f} days)", fontsize=16)
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "002_earth_moon_spacecraft_kdk_long_term.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
    return path


def main(dt=60., orbit_count=20.):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(orbit_count) or orbit_count <= 0:
        raise ValueError("dt and orbit_count must be finite and positive")
    positions, velocities, masses, gravitating, moon_period, spacecraft_period = initial_state()
    steps = max(1, int(orbit_count * moon_period / dt))
    duration = steps * dt
    print(f"KDK: {duration / moon_period:.6f} Earth-Moon periods, {duration / 86400:.2f} days; "
          f"dt={dt:g}/{dt / 2:g} s; steps={steps}/{2 * steps}", flush=True)
    check_passive_body(positions, velocities, masses, gravitating, dt)
    convergence = check_short_convergence(positions, velocities, masses, gravitating, dt, spacecraft_period)
    print(f"Passive-body check PASS; short-interval convergence ratio={convergence['ratio']:.4f} (target ~4)", flush=True)

    plots, summaries = [], []
    for factor in (1, 2):
        data = propagate(positions, velocities, masses, gravitating, dt / factor, steps * factor)
        compact, errors = analyze(data, positions, velocities, masses, moon_period, spacecraft_period)
        del data  # Release the full history before starting the next resolution.
        plots.append(compact)
        summaries.append({"dt_s": dt / factor, "steps": steps * factor, **errors})
        print(f"dt={dt / factor:g} s: bounded orbit / conservation PASS; "
              f"E={errors['energy']:.3e}, L={errors['angular_momentum']:.3e}, "
              f"P={errors['linear_momentum']:.3e}; "
              f"spacecraft r={errors['spacecraft_radius_min_m'] / 1000:.3f}.."
              f"{errors['spacecraft_radius_max_m'] / 1000:.3f} km", flush=True)

    final_difference = float(np.linalg.norm(plots[0]["final_relative_position"] - plots[1]["final_relative_position"]))
    print(f"Long-term spacecraft endpoint difference: {final_difference / 1000:.3f} km "
          "(phase sensitivity; bounded motion alone does not establish position accuracy)", flush=True)
    path = plot_results(*plots, dt, duration, moon_period, OUTPUT_DIR)
    summary = {"model": "ideal circular Earth/Moon + passive spacecraft; Newtonian point masses",
               "frame": "inertial, initial active barycenter at origin", "duration_s": duration,
               "moon_period_s": moon_period, "spacecraft_initial_period_s": spacecraft_period,
               "gravitating": gravitating.tolist(), "passive_body_check": "PASS",
               "short_interval_convergence": convergence, "long_runs": summaries,
               "long_spacecraft_endpoint_difference_m": final_difference,
               "note": "Short convergence and long stability do not certify long spacecraft phase accuracy."}
    summary_path = path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {path}\nSummary: {summary_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=60., help="Coarse dt in seconds; also runs dt/2")
    parser.add_argument("--orbits", type=float, default=20., help="Number of initial Earth-Moon periods")
    args = parser.parse_args()
    main(args.dt, args.orbits)
