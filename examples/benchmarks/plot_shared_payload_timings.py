#!/usr/bin/env python
"""Plot recent full-transport shared-payload reverse benchmark timings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "autodiff_transport_lagged_ntx" / "shared_payload_timings"


RUNS = {
    "bicgstab reference": {
        "runtime build": 302.582,
        "solver components": 120.784,
        "optimization smoke / setup": 120.948 + 50.454,
        "realized schedule forward": 729.535,
        "reverse segment sweep": 4374.090,
        "initial-Er root pullback": 162.307,
        "geometry final pullback": 21.938,
        "other / Python overhead": 1767.182,
    },
    "structured check": {
        "runtime build": 305.548,
        "solver components": 121.299,
        "optimization smoke / setup": 120.690 + 50.386,
        "realized schedule forward": 731.214,
        "reverse segment sweep": 4054.994,
        "initial-Er root pullback": 162.971,
        "geometry final pullback": 21.938,
        "other / Python overhead": 1751.429,
    },
}


SEGMENTS = {
    "bicgstab reference": [1136.419, 771.140, 379.218, 2087.311],
    "structured check": [1098.191, 731.620, 367.319, 1857.863],
}


COLORS = {
    "runtime build": "#2f6f9f",
    "solver components": "#73a2c6",
    "optimization smoke / setup": "#f2b134",
    "realized schedule forward": "#8ab17d",
    "reverse segment sweep": "#c44536",
    "initial-Er root pullback": "#7b5ea7",
    "geometry final pullback": "#4d908e",
    "other / Python overhead": "#b8b8b8",
}


def _write_csv(path: Path) -> None:
    labels = list(next(iter(RUNS.values())).keys())
    lines = ["run,component,seconds,minutes"]
    for run_name, values in RUNS.items():
        for label in labels:
            seconds = values[label]
            lines.append(f"{run_name},{label},{seconds:.6f},{seconds / 60.0:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_stacked(path: Path) -> None:
    labels = list(next(iter(RUNS.values())).keys())
    run_names = list(RUNS)
    y = np.arange(len(run_names))

    fig, ax = plt.subplots(figsize=(11.5, 4.8), dpi=180)
    left = np.zeros(len(run_names))
    for label in labels:
        values = np.array([RUNS[name][label] / 60.0 for name in run_names])
        ax.barh(
            y,
            values,
            left=left,
            height=0.56,
            label=label,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.8,
        )
        left += values

    totals = [sum(RUNS[name].values()) / 60.0 for name in run_names]
    for idx, total in enumerate(totals):
        ax.text(total + 1.0, idx, f"{total:.1f} min", va="center", fontsize=10)

    ax.set_yticks(y, run_names)
    ax.set_xlabel("Wall time [min]")
    ax.set_title("Full-Transport Shared-Payload Reverse Benchmark Timings")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_segments(path: Path) -> None:
    run_names = list(SEGMENTS)
    x = np.arange(4)
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    for idx, run_name in enumerate(run_names):
        offset = (idx - 0.5) * width
        ax.bar(
            x + offset,
            np.array(SEGMENTS[run_name]) / 60.0,
            width=width,
            label=run_name,
            edgecolor="white",
            linewidth=0.8,
        )

    ax.set_xticks(x, ["segment 1", "segment 2", "segment 3", "segment 4"])
    ax.set_ylabel("Wall time [min]")
    ax.set_title("Reverse Segmented Cotangent Sweep Timing")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(OUT_DIR / "shared_payload_timings.csv")
    _plot_stacked(OUT_DIR / "shared_payload_timings_stacked.png")
    _plot_segments(OUT_DIR / "shared_payload_reverse_segments.png")
    print(f"Wrote {OUT_DIR / 'shared_payload_timings.csv'}")
    print(f"Wrote {OUT_DIR / 'shared_payload_timings_stacked.png'}")
    print(f"Wrote {OUT_DIR / 'shared_payload_reverse_segments.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
