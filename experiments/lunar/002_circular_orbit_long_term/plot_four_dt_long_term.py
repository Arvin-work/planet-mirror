"""Old-style four-panel plot for KDK dt=60,30,1,0.5 s at one endpoint.

Read fine histories from the existing external run; regenerate only coarse
histories in bounded RAM blocks. Cache compact plotting data inside the project
so subsequent style changes require neither simulation nor a full disk scan.

Run from the project root:
    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/002_circular_orbit_long_term/plot_four_dt_long_term.py

Optional: --run-dir "Your temp data path, you need change by yourself"
          --rebuild-cache (otherwise reuse validated compact data)
"""
from pathlib import Path
import os
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORK = PROJECT / "work" / "002_kdk_disk_runtime"
for key, leaf in (("MPLCONFIGDIR", "mpl"), ("XDG_CACHE_HOME", "cache"),
                  ("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp")):
    (WORK / leaf).mkdir(parents=True, exist_ok=True)
    os.environ[key] = str(WORK / leaf)
os.environ["MPLBACKEND"] = "Agg"
sys.dont_write_bytecode = True

import argparse
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from nbody_kdk_long_term import initial_state, EARTH_RADIUS, analyze
from nbody_kdk_disk_long_term import (record_dtype, cached_force, fill_diagnostics,
                                     baseline, errors, check_block)
from orbit_sim.integrators import Velocity_Verlet_KDK

STEPS = (60., 30., 1., .5)
COLORS = ("#24864F", "#793BA0", "#2463B4", "#BF4B17")
STYLES = ("-", "--", "-.", ":")
MARKERS = ("o", "s", "^", "D")
NAME = "002_earth_moon_spacecraft_kdk_four_dt_long_term"
CACHE = PROJECT / "research" / "data" / f"{NAME}_plot_data.npz"
OUTPUT = PROJECT / "research" / "img"


def coarse_chunks(r0, v0, masses, mask, dt, steps, chunk_rows=200000):
    """Existing KDK and force; no full history retained or written to disk."""
    r, v = r0.copy(), v0.copy()
    force = cached_force(masses, mask)
    for start in range(0, steps + 1, chunk_rows):
        size = min(chunk_rows, steps + 1 - start)
        records = np.empty(size, dtype=record_dtype(len(masses)))
        for j in range(size):
            if start + j:
                r, v = Velocity_Verlet_KDK(r, v, dt, acceleration_function=force)
            records["positions"][j] = r
            records["velocities"][j] = v
        records["times"] = np.arange(start, start + size) * dt
        fill_diagnostics(records, masses, mask)
        yield records


def disk_chunks(folder):
    config = json.loads((folder / "parameters.json").read_text())
    status = json.loads((folder / "progress.json").read_text())
    paths = sorted(folder.glob("chunk_*.npy"))
    if not status["complete"] or len(paths) != status["chunks"]:
        raise ValueError(f"Incomplete source: {folder}")
    for i, path in enumerate(paths):
        if path.name != f"chunk_{i:06d}.npy":
            raise ValueError("Missing or reordered chunk")
        records = np.load(path, allow_pickle=False)
        if len(records) != min(config["chunk_rows"], config["steps"] + 1 - i * config["chunk_rows"]):
            raise ValueError("Incorrect chunk size")
        yield records
        if (i + 1) % 100 == 0:
            print(f"Read {folder.name}: {i + 1}/{len(paths)} blocks", flush=True)


def collect(chunks, dt, steps, r0, v0, masses, mask, period, local_period, max_samples=20000):
    """Reduce full histories to small plots, preserving every radius extremum.

    A local-period window can cross chunk boundaries. Its min/max accumulate
    into the same bin rather than treating each file as a new orbit window.
    """
    reference = baseline(r0, v0, masses, mask)
    sample_rows = np.unique(np.linspace(0, steps, min(steps + 1, max_samples), dtype=np.int64))
    sampled = np.empty(len(sample_rows), dtype=record_dtype(len(masses)))
    block = max(1, round(local_period / dt))
    bins = steps // block + 1
    lower, upper = np.full(bins, np.inf), np.full(bins, -np.inf)
    tail_start = max(0, steps - round(5 * local_period / dt))
    tails, maxima, start = [], {}, 0
    final = None
    for records in chunks:
        stop = start + len(records)
        if not len(records) or stop > steps + 1:
            raise ValueError("Incorrect source row count")
        np.testing.assert_array_equal(records["times"], np.arange(start, stop) * dt)
        check_block(records, reference, maxima)
        radius = np.linalg.norm(records["positions"][:, 2] - records["positions"][:, 0], axis=1)
        # Split only at the global window boundaries, including partial ends.
        boundaries = np.arange((start // block + 1) * block, stop, block) - start
        segments = np.r_[0, boundaries].astype(np.int64)
        ids = (start + segments) // block
        lower[ids] = np.minimum(lower[ids], np.minimum.reduceat(radius, segments))
        upper[ids] = np.maximum(upper[ids], np.maximum.reduceat(radius, segments))
        lo, hi = np.searchsorted(sample_rows, (start, stop))
        sampled[lo:hi] = records[sample_rows[lo:hi] - start]
        if stop > tail_start:
            first = max(start, tail_start) - start
            tails.append((records["positions"][first:, 2] - records["positions"][first:, 0]).copy())
        final = records[-1].copy()
        start = stop
    if start != steps + 1:
        raise ValueError("History ends before the requested common endpoint")
    e, l, p, _, _ = errors(sampled, reference)
    tail = np.concatenate(tails)
    tail_indices = np.unique(np.linspace(0, len(tail) - 1, min(12000, len(tail)), dtype=np.int64))
    starts = np.arange(bins) * block
    ends = np.minimum(starts + block - 1, steps)
    result = {"positions": sampled["positions"].copy(),
              "time_in_moon_periods": sampled["times"] / period,
              "energy_error": e, "angular_error": l, "momentum_error": p,
              "spacecraft_tail": tail[tail_indices],
              "final_positions": final["positions"].copy(),
              "final_relative_position": final["positions"][2] - final["positions"][0],
              "envelope_time": (starts + ends) * dt / (2 * period),
              "radius_min_km": lower / 1000, "radius_max_km": upper / 1000}
    if not all(np.all(np.isfinite(a)) for a in result.values()):
        raise ValueError("Nonfinite compact plotting data")
    return result, maxima


def source_config(run_dir, r, v, masses, mask):
    configs, checksums = {}, {}
    for name, dt in (("dt_1s", 1.), ("dt_0p5s", .5)):
        folder = run_dir / name
        for filename in ("parameters.json", "progress.json"):
            p = folder / filename
            checksums[f"{name}/{filename}"] = hashlib.sha256(p.read_bytes()).hexdigest()
        config = json.loads((folder / "parameters.json").read_text())
        if config["dt_s"] != dt:
            raise ValueError("Unexpected source time step")
        for key, value in (("positions", r), ("velocities", v), ("masses_kg", masses), ("gravitating", mask)):
            np.testing.assert_array_equal(config[key], value)
        if not json.loads((folder / "progress.json").read_text())["complete"]:
            raise ValueError("Fine trajectory is not complete")
        configs[dt] = config
    duration = configs[1.]["steps"]
    if configs[.5]["steps"] * .5 != duration or any(duration % dt for dt in STEPS):
        raise ValueError("Time steps do not share the expected common endpoint")
    key = json.dumps({"schema": 1, "source": str(run_dir.resolve()), "checksums": checksums,
                      "duration": duration, "coarse_dt": [60., 30.]}, sort_keys=True)
    return int(duration), hashlib.sha256(key.encode()).hexdigest()


def load_or_collect(run_dir, rebuild=False):
    r, v, masses, mask, period, local_period = initial_state()
    duration, key = source_config(run_dir, r, v, masses, mask)
    if CACHE.exists() and not rebuild:
        with np.load(CACHE, allow_pickle=False) as cache:
            metadata = json.loads(str(cache["metadata"]))
            if metadata["source_key"] == key:
                results = [{name.split("__", 1)[1]: cache[name].copy()
                            for name in cache.files if name.startswith(f"g{i}__")}
                           for i in range(len(STEPS))]
                print(f"Reused compact plotting data: {CACHE}", flush=True)
                return results, metadata
    results, metrics = [], []
    for dt in STEPS:
        steps = int(duration / dt)
        if dt in (60., 30.):
            print(f"Recomputing dt={dt:g}s in memory ({steps:,} steps)", flush=True)
            chunks = coarse_chunks(r, v, masses, mask, dt, steps)
        else:
            folder = run_dir / ("dt_1s" if dt == 1. else "dt_0p5s")
            print(f"Reading existing dt={dt:g}s history; no integration", flush=True)
            chunks = disk_chunks(folder)
        compact, maximum = collect(chunks, dt, steps, r, v, masses, mask, period, local_period)
        results.append(compact)
        metrics.append({"dt_s": dt, "steps": steps, **maximum})
        print(f"dt={dt:g}s ready; radius={maximum['spacecraft_radius_min_m'] / 1000:.6f}.."
              f"{maximum['spacecraft_radius_max_m'] / 1000:.6f} km", flush=True)
    # The coarse regenerated trajectories must reproduce the old summary.
    old_path = OUTPUT / "002_earth_moon_spacecraft_kdk_long_term.json"
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if old["duration_s"] != duration:
            raise ValueError("Old figure used a different endpoint")
        for new, expected in zip(metrics[:2], old["long_runs"]):
            for field in ("spacecraft_radius_min_m", "spacecraft_radius_max_m"):
                np.testing.assert_allclose(new[field], expected[field], rtol=0, atol=1.e-6)
    metadata = {"source_key": key, "duration_s": duration, "moon_period_s": period,
                "spacecraft_initial_period_s": local_period, "source_directory": str(run_dir),
                "metrics": metrics, "radius_envelope_uses_all_snapshots": True,
                "orbit_and_diagnostic_lines_max_samples": 20000,
                "note": "Envelope is per-local-period radial min/max, not uncertainty or position error."}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"g{i}__{name}": value for i, result in enumerate(results) for name, value in result.items()}
    temporary = CACHE.with_suffix(".npz.partial")
    with temporary.open("wb") as f:
        np.savez_compressed(f, metadata=np.asarray(json.dumps(metadata)), **arrays)
    temporary.replace(CACHE)
    print(f"Saved reusable plot data: {CACHE}", flush=True)
    return results, metadata


def plot_results(results, metadata, output=OUTPUT):
    """Keep the reference four-panel layout; add a fine-step radius zoom."""
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        orbit, local, diagnostics, radius = axes.flat
        body_colors = ("#2463B4", "#C69A1A", "#24864F")
        for result, dt, color, style, marker in zip(results, STEPS, COLORS, STYLES, MARKERS):
            for body, body_color in enumerate(body_colors):
                track = result["positions"][:, body] / 1000
                orbit.plot(track[:, 0], track[:, 1], color=body_color, linestyle=style, linewidth=1)
                final = result["final_positions"][body] / 1000
                orbit.scatter(*final[:2], color=body_color, marker=marker, s=25, zorder=5)
            tail = result["spacecraft_tail"] / 1000
            local.plot(tail[:, 0], tail[:, 1], color=color, linestyle=style, linewidth=1.2,
                       label=f"dt = {dt:g} s")
            local.scatter(*tail[-1, :2], color=color, marker=marker, s=35, zorder=5)
            for field, quantity_style in (("energy_error", "-"), ("angular_error", "--"), ("momentum_error", ":")):
                diagnostics.semilogy(result["time_in_moon_periods"], np.maximum(result[field], 1e-18),
                                      color=color, linestyle=quantity_style, linewidth=1)
            radius.fill_between(result["envelope_time"], result["radius_min_km"], result["radius_max_km"],
                                color=color, alpha=.2, label=f"dt = {dt:g} s")
            radius.plot(result["envelope_time"], result["radius_max_km"], color=color, linewidth=.7)
        bodies = [Line2D([], [], color=c, label=n) for c, n in zip(body_colors, ("Earth", "Moon", "Spacecraft"))]
        time_styles = [Line2D([], [], color="black", linestyle=s, marker=m, markersize=4, label=f"dt = {dt:g} s")
                       for dt, s, m in zip(STEPS, STYLES, MARKERS)]
        body_legend = orbit.legend(handles=bodies, fontsize=8, loc="upper left", title="Bodies")
        orbit.add_artist(body_legend)
        orbit.legend(handles=time_styles, fontsize=8, loc="lower left", title="Time Step")
        orbit.scatter(0, 0, color="black", marker="+", s=30)
        orbit.set(title="Barycentric Trajectories (markers: final states)", xlabel="x (km)", ylabel="y (km)")
        orbit.axis("equal")
        local.add_patch(plt.Circle((0, 0), EARTH_RADIUS / 1000, color=body_colors[0], alpha=.2))
        local.set(title="Spacecraft Relative to Earth (last ~5 local periods)",
                  xlabel="Relative x (km)", ylabel="Relative y (km)")
        local.axis("equal")
        local.legend(fontsize=8, loc="upper right")
        time_colors = [Line2D([], [], color=c, label=f"dt = {dt:g} s") for dt, c in zip(STEPS, COLORS)]
        quantities = [Line2D([], [], color="black", linestyle=s, label=n)
                      for s, n in (("-", "Energy"), ("--", "Angular momentum"), (":", "Linear momentum"))]
        dt_legend = diagnostics.legend(handles=time_colors, fontsize=8, loc="upper left", ncol=2)
        diagnostics.add_artist(dt_legend)
        diagnostics.legend(handles=quantities, fontsize=8, loc="lower right")
        diagnostics.set(title="Earth + Moon Conservation (active bodies only)",
                        xlabel="Time / initial Earth-Moon period", ylabel="Normalized absolute error")
        radius.axhline(10000., color="black", linestyle=":", linewidth=1)
        radius.set(title="Spacecraft Distance from Earth's Center (local-orbit min/max)",
                   xlabel="Time / initial Earth-Moon period", ylabel="Distance (km)")
        radius.ticklabel_format(axis="y", style="plain", useOffset=False)
        radius.legend(fontsize=8, loc="upper right", ncol=2)
        # The 60/30 s kilometer-scale ranges otherwise hide the fine meter-scale
        # ranges. An inset shows only 1/0.5 s and uses explicit kilometer ticks.
        zoom = radius.inset_axes([.48, .28, .47, .36])
        for result, dt, color in zip(results[2:], STEPS[2:], COLORS[2:]):
            zoom.fill_between(result["envelope_time"], result["radius_min_km"], result["radius_max_km"],
                              color=color, alpha=.25)
            zoom.plot(result["envelope_time"], result["radius_max_km"], color=color, linewidth=.7)
        zoom.axhline(10000., color="black", linestyle=":", linewidth=.7)
        zoom.set_title("1 s / 0.5 s zoom", fontsize=8)
        zoom.ticklabel_format(axis="y", style="plain", useOffset=False)
        zoom.tick_params(labelsize=7)
        zoom.grid()
        for ax in axes.flat:
            ax.grid()
        fig.suptitle(f"Earth-Moon-Spacecraft: Long-term KDK ({metadata['duration_s'] / 86400:.1f} days)", fontsize=16)
        fig.tight_layout()
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{NAME}.png"
        fig.savefig(path, dpi=300)
        return fig, path


def main(run_dir=None, rebuild=False):
    if run_dir is None:
        launch = json.loads((WORK / "launch.json").read_text())
        run_dir = Path(launch["run_directory"])
    results, metadata = load_or_collect(Path(run_dir), rebuild)
    fig, image = plot_results(results, metadata)
    plt.close(fig)
    summary = dict(metadata)
    summary["plot_data_cache"] = str(CACHE)
    summary["final_position_differences_to_dt0p5_m"] = {
        f"{dt:g}": float(np.linalg.norm(result["final_relative_position"] - results[-1]["final_relative_position"]))
        for dt, result in zip(STEPS, results)}
    (OUTPUT / f"{NAME}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved: {image}\nCompact cache: {CACHE.stat().st_size / 1e6:.3f} MB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    main(args.run_dir, args.rebuild_cache)
