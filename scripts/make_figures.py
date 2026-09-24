"""Redraw docs/error_rate.png from results/error_rate_sweep.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sweep = json.loads((ROOT / "results" / "error_rate_sweep.json").read_text())
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for name, rows in sweep.items():
        if name.startswith("sandbox repair + verified"):
            continue  # identical to the sandbox strategy on this plot
        rates = [100 * r["error_rate"] for r in rows]
        left.plot(rates, [100 * r["success_rate"] for r in rows], marker="o", label=name)
        right.plot(rates, [r["harmful_per_100_tasks"] for r in rows], marker="o", label=name)
    left.set(xlabel="plans the planner corrupts (%)", ylabel="tasks completed (%)", title="Success")
    right.set(
        xlabel="plans the planner corrupts (%)", ylabel="per 100 tasks", title="Irreversible side effects left behind"
    )
    for ax in (left, right):
        ax.grid(alpha=0.3)
    right.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "error_rate.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
