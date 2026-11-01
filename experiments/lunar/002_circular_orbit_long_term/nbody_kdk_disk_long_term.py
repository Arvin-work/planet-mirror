"""Disk-backed 1 s / 0.5 s KDK comparison; preserve all prior experiments.

Full histories go ONLY to --data-root (default: the authorized external disk).
Project-local work/ holds caches; research/img/ holds final figures and reports.
Each .npy chunk contains every step as a structured float64 record.
"""
import os
from pathlib import Path
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
import csv
from datetime import datetime, timezone
import json
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np

from nbody_kdk_long_term import initial_state, EARTH_RADIUS, SPACECRAFT_RADIUS, EARTH_MOON_DISTANCE
from orbit_sim.dynamics import G, make_nbody_acceleration
from orbit_sim.integrators import Velocity_Verlet_KDK
from orbit_sim.diagnostics import (total_energy, total_linear_momentum,
                                  total_angular_momentum, center_of_mass)

EXTERNAL_ROOT = Path("unknown") # This part need to change your data root path
OUTPUT = PROJECT / "research" / "img"
LIMIT = 100_000_000_000  # decimal bytes, including all files in this run
PREFIX = "002_earth_moon_spacecraft_kdk_dt1_vs_dt0p5"


def record_dtype(n):
    return np.dtype([("times", "<f8"), ("positions", "<f8", (n, 3)),
                     ("velocities", "<f8", (n, 3)), ("total_energy", "<f8"),
                     ("total_linear_momentum", "<f8", (3,)),
                     ("total_angular_momentum", "<f8", (3,)),
                     ("center_of_mass", "<f8", (3,))])


def write_json(path, value):
    """Atomic replace; the temporary file stays in the same allowed directory."""
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def tree_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def cached_force(masses, mask):
    """Reuse the unchanged KDK force between adjacent steps (same snapshot).

    Metadata are frozen by the public force factory. A changed snapshot always
    calls the public N-body force. This changes no integration equations.
    """
    force = make_nbody_acceleration(masses, mask)
    previous = acceleration = None
    def evaluate(r):
        nonlocal previous, acceleration
        if previous is None or not np.array_equal(r, previous):
            acceleration = force(r)
            previous = r.copy()
        return acceleration
    return evaluate


def fill_diagnostics(records, masses, mask):
    """Batch equivalents of the public active-body diagnostics; no integration."""
    selected = mask & (masses > 0)
    r = records["positions"][:, selected]
    v = records["velocities"][:, selected]
    m = masses[selected]
    records["total_energy"] = .5 * np.sum(m[None, :, None] * v**2, axis=(1, 2))
    for i in range(len(m)):
        distance = np.linalg.norm(r[:, i + 1:] - r[:, i, None], axis=2)
        if np.any(distance == 0):
            raise ValueError("Coincident active bodies")
        records["total_energy"] -= np.sum(G * m[i] * m[i + 1:] / distance, axis=1)
    records["total_linear_momentum"] = np.sum(m[None, :, None] * v, axis=1)
    records["total_angular_momentum"] = np.sum(m[None, :, None] * np.cross(r, v), axis=1)
    records["center_of_mass"] = np.sum(m[None, :, None] * r, axis=1) / m.sum()


def baseline(r, v, m, mask):
    return {"energy": float(total_energy(r, v, m, mask)),
            "angular": total_angular_momentum(r, v, m, mask).tolist(),
            "momentum": total_linear_momentum(v, m, mask).tolist(),
            "com": center_of_mass(r, m, mask).tolist(),
            "mass": float(m[mask].sum()),
            "momentum_scale": float(np.sum(m[mask] * np.linalg.norm(v[mask], axis=1)))}


def errors(records, reference):
    e = np.abs((records["total_energy"] - reference["energy"]) / abs(reference["energy"]))
    l = np.linalg.norm(records["total_angular_momentum"] - reference["angular"], axis=1)
    l /= np.linalg.norm(reference["angular"])
    p = np.linalg.norm(records["total_linear_momentum"] - reference["momentum"], axis=1)
    p /= reference["momentum_scale"]
    expected = (np.asarray(reference["com"]) + records["times"][:, None]
                * np.asarray(reference["momentum"]) / reference["mass"])
    com = np.linalg.norm(records["center_of_mass"] - expected, axis=1)
    rho = records["positions"][:, 2] - records["positions"][:, 0]
    radius = np.linalg.norm(rho, axis=1)
    return e, l, p, com, radius


def check_block(records, reference, maxima):
    # A structured record cannot be directly passed to np.isfinite.
    if not all(np.all(np.isfinite(records[name])) for name in records.dtype.names):
        raise ValueError("Nonfinite history")
    e, l, p, com, radius = errors(records, reference)
    values = {"energy": float(e.max()), "angular_momentum": float(l.max()),
              "linear_momentum": float(p.max()), "center_of_mass_drift_m": float(com.max()),
              "spacecraft_radius_max_m": float(radius.max())}
    for key, value in values.items():
        maxima[key] = max(maxima.get(key, 0.), value)
    maxima["spacecraft_radius_min_m"] = min(maxima.get("spacecraft_radius_min_m", float("inf")),
                                            float(radius.min()))
    assert maxima["energy"] < 1.e-6, maxima
    assert max(maxima["angular_momentum"], maxima["linear_momentum"]) < 1.e-10, maxima
    assert maxima["center_of_mass_drift_m"] / EARTH_MOON_DISTANCE < 1.e-10, maxima
    assert radius.min() > EARTH_RADIUS
    assert np.max(np.abs(radius / SPACECRAFT_RADIUS - 1)) < .1


def run_group(folder, r0, v0, masses, mask, dt, steps, chunk_rows=200000,
              run_root=None, limit=LIMIT, quiet=False):
    """Write every snapshot in bounded chunks, with restart checkpoints.

    Files cover contiguous row indices; row k has t=k*dt, including row 0.
    A completed checkpoint may be resumed with identical parameters. An orphan
    file after an interrupted rename/checkpoint is rejected rather than overwritten.
    """
    if steps < 0 or dt <= 0 or chunk_rows < 1:
        raise ValueError("Invalid step/count parameters")
    run_root = run_root or folder
    folder.mkdir(parents=True, exist_ok=True)
    config = {"dt_s": dt, "steps": steps, "chunk_rows": chunk_rows,
              "positions": r0.tolist(), "velocities": v0.tolist(),
              "masses_kg": masses.tolist(), "gravitating": mask.tolist()}
    config_path, status_path = folder / "parameters.json", folder / "progress.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Resume parameters differ")
        status = json.loads(status_path.read_text())
    else:
        write_json(config_path, config)
        status = {"next_row": 0, "chunks": 0, "maxima": {},
                  "last_positions": r0.tolist(), "last_velocities": v0.tolist(), "complete": False}
        write_json(status_path, status)
    if len(list(folder.glob("chunk_*.npy"))) != status["chunks"]:
        raise ValueError("Chunk/checkpoint mismatch; inspect the run before resuming")
    if status["complete"]:
        return status
    r, v = np.asarray(status["last_positions"]), np.asarray(status["last_velocities"])
    force, reference = cached_force(masses, mask), baseline(r0, v0, masses, mask)
    dtype = record_dtype(len(masses))
    occupied = tree_bytes(run_root)
    clock_start = time.perf_counter()
    initial_row = status["next_row"]
    last_print = clock_start
    while status["next_row"] <= steps:
        start = status["next_row"]
        count = min(chunk_rows, steps + 1 - start)
        records = np.empty(count, dtype=dtype)
        for j in range(count):
            row = start + j
            if row:
                r, v = Velocity_Verlet_KDK(r, v, dt, acceleration_function=force)
            records["positions"][j] = r
            records["velocities"][j] = v
        records["times"] = np.arange(start, start + count) * dt
        fill_diagnostics(records, masses, mask)
        check_block(records, reference, status["maxima"])
        required = records.nbytes + 4096
        if occupied + required > limit:
            raise RuntimeError("100 GB run limit would be exceeded; completed chunks retained")
        if shutil.disk_usage(folder).free < required + 100_000_000:
            raise RuntimeError("Insufficient free disk space; completed chunks retained")
        path = folder / f"chunk_{status['chunks']:06d}.npy"
        temporary = path.with_suffix(".partial")
        with temporary.open("xb") as stream:
            np.save(stream, records, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(path)
        temporary.replace(path)
        occupied += path.stat().st_size
        status.update(next_row=start + count, chunks=status["chunks"] + 1,
                      last_positions=r.tolist(), last_velocities=v.tolist(),
                      complete=start + count == steps + 1)
        write_json(status_path, status)
        now = time.perf_counter()
        if not quiet and (now - last_print > 300 or status["complete"]):
            rate = (status["next_row"] - initial_row) / max(now - clock_start, 1.e-9)
            print(f"dt={dt:g}s: {100 * status['next_row'] / (steps + 1):.2f}% "
                  f"({status['next_row'] - 1:,}/{steps:,} steps), {rate:.0f} steps/s; "
                  f"run files={occupied / 1e9:.3f} GB", flush=True)
            last_print = now
        del records
    return status


def read_rows(folder, indices):
    """Read selected rows without loading the complete history."""
    config = json.loads((folder / "parameters.json").read_text())
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices > config["steps"]):
        raise IndexError("History row out of range")
    out = np.empty(len(indices), dtype=record_dtype(len(config["masses_kg"])))
    chunk_ids = indices // config["chunk_rows"]
    for chunk in np.unique(chunk_ids):
        selected = chunk_ids == chunk
        data = np.load(folder / f"chunk_{chunk:06d}.npy", mmap_mode="r", allow_pickle=False)
        out[selected] = data[indices[selected] % config["chunk_rows"]]
        del data
    return out


def compare_and_plot(run_root, coarse_steps, dt, period, output=OUTPUT):
    """Compare ALL matching times blockwise, then plot sampled differences."""
    coarse_folder, fine_folder = run_root / "dt_1s", run_root / "dt_0p5s"
    config = json.loads((coarse_folder / "parameters.json").read_text())
    samples = np.unique(np.linspace(0, coarse_steps, min(coarse_steps + 1, 20000), dtype=np.int64))
    maximum_difference, squared_sum, count = 0., 0., 0
    for path in sorted(coarse_folder.glob("chunk_*.npy")):
        index = int(path.stem.split("_")[-1])
        coarse = np.load(path, allow_pickle=False)
        rows = np.arange(index * config["chunk_rows"], index * config["chunk_rows"] + len(coarse))
        fine = read_rows(fine_folder, 2 * rows)
        np.testing.assert_array_equal(coarse["times"], fine["times"])
        rho_coarse = coarse["positions"][:, 2] - coarse["positions"][:, 0]
        rho_fine = fine["positions"][:, 2] - fine["positions"][:, 0]
        difference = np.linalg.norm(rho_coarse - rho_fine, axis=1)
        maximum_difference = max(maximum_difference, float(difference.max()))
        squared_sum += float(np.sum(difference**2))
        count += len(difference)
    coarse, fine = read_rows(coarse_folder, samples), read_rows(fine_folder, 2 * samples)
    rho_coarse = coarse["positions"][:, 2] - coarse["positions"][:, 0]
    rho_fine = fine["positions"][:, 2] - fine["positions"][:, 0]
    difference = np.linalg.norm(rho_coarse - rho_fine, axis=1)
    radial_difference = np.linalg.norm(rho_coarse, axis=1) - np.linalg.norm(rho_fine, axis=1)
    angle = np.unwrap(np.arctan2(np.cross(rho_coarse, rho_fine)[:, 2],
                                np.sum(rho_coarse * rho_fine, axis=1)))
    reference = baseline(*[np.asarray(config[key]) for key in ("positions", "velocities", "masses_kg", "gravitating")])
    output.mkdir(parents=True, exist_ok=True)
    with plt.style.context("default"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        x = coarse["times"] / period
        axes[0, 0].plot(x, difference / 1000, color="#2463B4")
        axes[0, 0].set(title="Spacecraft Position Difference: 1 s vs 0.5 s",
                       xlabel="Time / initial Earth-Moon period", ylabel="Relative-to-Earth position difference (km)")
        axes[0, 1].plot(x, np.degrees(angle), color="#793BA0")
        axes[0, 1].set(title="Spacecraft Phase Difference (fine minus coarse)",
                       xlabel="Time / initial Earth-Moon period", ylabel="Angle difference (degrees)")
        axes[1, 0].plot(x, radial_difference, color="#24864F")
        axes[1, 0].set(title="Spacecraft Radial Difference (coarse minus fine)",
                       xlabel="Time / initial Earth-Moon period", ylabel="Distance difference (m)")
        for records, step, style in ((coarse, dt, "--"), (fine, dt / 2, "-")):
            e, l, p, _, _ = errors(records, reference)
            for values, label, color in ((e, "E", "#2463B4"), (l, "L", "#24864F"), (p, "P", "#793BA0")):
                axes[1, 1].semilogy(x, np.maximum(values, 1e-18), color=color,
                                    linestyle=style, linewidth=1, label=f"{label}, dt={step:g}s")
        axes[1, 1].set(title="Earth + Moon Conservation", xlabel="Time / initial Earth-Moon period",
                       ylabel="Normalized absolute error")
        axes[1, 1].legend(fontsize=8, ncol=2)
        for ax in axes.flat:
            ax.grid()
        fig.suptitle(f"KDK Step-size Comparison: 1 s / 0.5 s ({coarse['times'][-1] / 86400:.1f} days)", fontsize=16)
        fig.tight_layout()
        image = output / f"{PREFIX}.png"
        fig.savefig(image, dpi=300)
        plt.close(fig)
    summary = {"duration_s": float(coarse["times"][-1]), "matching_samples_checked": count,
               "final_spacecraft_position_difference_m": float(difference[-1]),
               "maximum_spacecraft_position_difference_m": maximum_difference,
               "rms_spacecraft_position_difference_m": float(np.sqrt(squared_sum / count)),
               "final_phase_difference_degrees": float(np.degrees(angle[-1])),
               "data_directory": str(run_root),
               "note": "Two resolutions measure agreement, not error against an exact solution or proof of convergence."}
    for key, folder in (("coarse", coarse_folder), ("fine", fine_folder)):
        summary[key] = json.loads((folder / "progress.json").read_text())["maxima"]
    write_json(output / f"{PREFIX}.json", summary)
    return image, summary


def main(data_root=EXTERNAL_ROOT, resume=None, duration=None, chunk_rows=200000, output=OUTPUT):
    for name in ("mpl", "cache", "tmp"):
        (WORK / name).mkdir(parents=True, exist_ok=True)
    r, v, masses, mask, moon_period, _ = initial_state()
    duration = int(20 * moon_period / 60) * 60 if duration is None else duration
    if not np.isfinite(duration) or duration <= 0 or duration != int(duration):
        raise ValueError("Duration must be a positive whole number of seconds")
    steps = int(duration)
    data_root = Path(data_root).resolve()
    output = Path(output).resolve()
    if data_root != EXTERNAL_ROOT.resolve() and not data_root.is_relative_to(PROJECT):
        raise ValueError("Data writes are limited to the specified external directory or project")
    if not output.is_relative_to(PROJECT):
        raise ValueError("Images/reports must stay within the project")
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data directory must already exist: {data_root}")
    projected = (3 * steps + 2) * record_dtype(3).itemsize + 10_000_000
    if projected > LIMIT or shutil.disk_usage(data_root).free < projected + 1_000_000_000:
        raise RuntimeError("Projected histories exceed the 100 GB limit or available space")
    if resume is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_root = data_root / f"002_kdk_dt1_long_term_{stamp}"
        run_root.mkdir(exist_ok=False)
    else:
        run_root = Path(resume).resolve()
        if run_root.parent != data_root or not run_root.is_dir():
            raise ValueError("Resume folder must be directly under data-root")
    print(f"Full-history directory: {run_root}\nProjected data: {projected / 1e9:.3f} GB; "
          f"duration={duration / 86400:.2f} days", flush=True)
    write_json(run_root / "run_parameters.json", {"duration_s": duration, "chunk_rows": chunk_rows,
                                                "projected_bytes": projected, "limit_bytes": LIMIT,
                                                "moon_period_s": moon_period, "units": "m, s, kg"})
    for name, dt, count in (("dt_1s", 1., steps), ("dt_0p5s", .5, 2 * steps)):
        run_group(run_root / name, r, v, masses, mask, dt, count, chunk_rows, run_root)
    image, summary = compare_and_plot(run_root, steps, 1., moon_period, output)
    files = sorted(p for p in run_root.rglob("*") if p.is_file())
    with (output / f"{PREFIX}_files.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("path", "bytes"))
        for p in files:
            writer.writerow((str(p), p.stat().st_size))
    summary["external_file_count"] = len(files)
    summary["external_bytes"] = tree_bytes(run_root)
    assert summary["external_bytes"] < LIMIT
    write_json(output / f"{PREFIX}.json", summary)
    print(f"Saved: {image}\nExternal files: {len(files)}, {summary['external_bytes'] / 1e9:.3f} GB; "
          f"final spacecraft difference={summary['final_spacecraft_position_difference_m'] / 1000:.6f} km", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=EXTERNAL_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--duration", type=int, help="Short verification interval; default matches prior 545.7-day run")
    parser.add_argument("--chunk-rows", type=int, default=200000)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT, help="Project-local output; isolate short verification figures")
    args = parser.parse_args()
    main(args.data_root, args.resume, args.duration, args.chunk_rows, args.output_dir)
